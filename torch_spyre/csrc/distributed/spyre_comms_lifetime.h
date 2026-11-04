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
#pragma once

#include <cstdint>
#include <memory>
#include <spyre_comms.hpp>

namespace spyre {

/*
 * Single owner of the spyre_comms initialize_library()/finalize_library()
 * reference count for torch-spyre.
 *
 * Every initialize_library() performed by torch-spyre goes through this
 * module, so that every reference taken is released exactly once, and so that
 * nothing is ever finalized when spyre_comms was never initialized.
 */

/**
 * Take one spyre_comms reference (initializing the library if needed).
 * @return an epoch token to pass to release_comms().
 */
uint64_t acquire_comms();

/**
 * Release a reference taken by acquire_comms(). No-op if shutdown_all() has
 * already released it (stale epoch). The cached collective plans are dropped
 * before the final finalize_library().
 */
void release_comms(uint64_t epoch);

/**
 * Release every outstanding reference. Safe to call when spyre_comms was never
 * initialized (no-op). Used for process-exit cleanup.
 */
void shutdown_all();

/**
 * Return the spyre_comms world context for the compiled-collective plan ops,
 * initializing the library if no reference is currently held.
 */
std::shared_ptr<spyre_comms::Context> ensure_context();

/**
 * Drop all cached collective plans (WorkScheduleInfo). Defined in
 * spyre_distributed.cpp. Must be called before the final finalize_library().
 */
void clear_wsi_cache();

}  // namespace spyre
