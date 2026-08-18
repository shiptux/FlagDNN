# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Resolve FLAGDNN_BACKEND_AUTO without making the result depend on backend
# registration order. NVIDIA remains the compatibility default whenever it is
# part of a multi-backend build; otherwise the first configured plugin is used.
# An explicit selection may name a plugin supplied at runtime through
# FLAGDNN_BACKEND_PATH. This keeps core-only SDKs extensible without weakening
# the deterministic AUTO policy for plugins built in this tree.
function(flagdnn_resolve_default_backend output configured_backends selection)
  if(NOT selection MATCHES "^(auto|[a-z][a-z0-9_]*)$")
    message(FATAL_ERROR
      "FLAGDNN_DEFAULT_BACKEND must be auto or match [a-z][a-z0-9_]*")
  endif()

  set(_flagdnn_configured_backends ${configured_backends})
  if(selection STREQUAL "auto")
    list(FIND _flagdnn_configured_backends nvidia _flagdnn_nvidia_index)
    if(NOT _flagdnn_nvidia_index EQUAL -1)
      set(_flagdnn_effective_backend nvidia)
    elseif(_flagdnn_configured_backends)
      list(GET _flagdnn_configured_backends 0 _flagdnn_effective_backend)
    else()
      # Core-only builds still need a deterministic compiled-in fallback.
      set(_flagdnn_effective_backend nvidia)
    endif()
  else()
    set(_flagdnn_effective_backend "${selection}")
  endif()

  set(${output} "${_flagdnn_effective_backend}" PARENT_SCOPE)
endfunction()
