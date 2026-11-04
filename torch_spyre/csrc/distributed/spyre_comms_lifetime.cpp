/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "spyre_comms_lifetime.h"

#include <c10/util/Exception.h>

#include <mutex>

#include "../logging.h"
#include "../module.h"
#include "../spyre_stream.h"

namespace spyre {

namespace {

std::mutex lifetime_mutex;
int outstanding_refs = 0;    // spyre_comms references held by torch-spyre
uint64_t current_epoch = 0;  // bumped by shutdown_all()

// Must be called with lifetime_mutex held.
void init_locked() {
  spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                  spyre::getDefaultStreamRuntimeHandle());
  ++outstanding_refs;
}

// Must be called with lifetime_mutex held and outstanding_refs > 0.
void finalize_locked() {
  if (--outstanding_refs == 0) {
    // Plans reference the world context; drop them before it goes away.
    clear_wsi_cache();
  }
  spyre_comms::finalize_library();
}

}  // namespace

uint64_t acquire_comms() {
  std::lock_guard<std::mutex> lock(lifetime_mutex);
  init_locked();
  return current_epoch;
}

void release_comms(uint64_t epoch) {
  std::lock_guard<std::mutex> lock(lifetime_mutex);
  if (epoch != current_epoch || outstanding_refs == 0) {
    return;
  }
  finalize_locked();
}

void shutdown_all() {
  std::lock_guard<std::mutex> lock(lifetime_mutex);
  if (outstanding_refs == 0) {
    return;  // spyre_comms was never initialized (or is already finalized)
  }
  SPYRE_RUNTIME_DEBUG() << "Releasing " << outstanding_refs
                        << " outstanding spyre-comms reference(s)";
  ++current_epoch;  // invalidate tokens held by live backends
  while (outstanding_refs > 0) {
    finalize_locked();
  }
}

std::shared_ptr<spyre_comms::Context> ensure_context() {
  {
    std::lock_guard<std::mutex> lock(lifetime_mutex);
    if (outstanding_refs == 0) {
      // Spyre Comms is initialised with the default Flex stream handle.
      // Because all collective operations are injected into that same stream,
      // Flex enforces ordering between comms operations and any surrounding
      // compute kernels.  This data-dependency ordering is what makes skipping
      // work->wait() in spyre_wait_work_impl valid: the device itself
      // serialises the collective result before any subsequent operation on
      // the stream can consume it.  Torch Spyre calls synchronize() when
      // exposing results to the user, so there is no gap at the end of the
      // graph either.
      SPYRE_RUNTIME_DEBUG() << "Initializing spyre-comms library";
      init_locked();
    }
  }
  auto context = spyre_comms::get_world_context();
  TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  return context;
}

}  // namespace spyre
