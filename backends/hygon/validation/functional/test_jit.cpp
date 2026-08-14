/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <Python.h>

#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fe = ::flagdnn_frontend;
constexpr const char *kPoisonedHelperProbe = "--poison-gen-ssig";

void check_frontend(fe::error_t status, const char *operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + status.get_message());
  }
}

class TemporaryCache {
public:
  TemporaryCache() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "flagdnn-hygon-add-jit-build-XXXXXX")
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

fe::graph::Graph::Tensor make_tensor(fe::graph::Graph &graph, const char *name,
                                     std::int64_t uid) {
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_name(name)
                          .set_uid(uid)
                          .set_data_type(fe::DataType_t::FLOAT)
                          .set_dim({1024})
                          .set_stride({1}));
}

void preload_wrong_gen_ssig(const char *origin) {
  Py_InitializeEx(0);
  if (!Py_IsInitialized()) {
    throw std::runtime_error("cannot initialize probe Python interpreter");
  }
  PyObject *module = PyModule_New("gen_ssig");
  PyObject *file = PyUnicode_DecodeFSDefault(origin);
  PyObject *modules = PyImport_GetModuleDict();
  if (module == nullptr || file == nullptr || modules == nullptr ||
      PyObject_SetAttrString(module, "__file__", file) != 0 ||
      PyDict_SetItemString(modules, "gen_ssig", module) != 0) {
    PyErr_Clear();
    Py_XDECREF(file);
    Py_XDECREF(module);
    throw std::runtime_error("cannot preload poisoned gen_ssig probe module");
  }
  Py_DECREF(file);
  Py_DECREF(module);
}

void run_poisoned_helper_probe(char **argv) {
  const pid_t child = fork();
  if (child < 0) {
    throw std::runtime_error("fork for poisoned helper probe failed");
  }
  if (child == 0) {
    execl(argv[0], argv[0], argv[1], argv[2], kPoisonedHelperProbe,
          static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    throw std::runtime_error(
        "Hygon JIT did not reject a preloaded wrong gen_ssig before use");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool poisoned_helper_probe =
        argc == 4 && std::string_view(argv[3]) == kPoisonedHelperProbe;
    if (argc != 3 && !poisoned_helper_probe) {
      std::cerr << "usage: native_hygon_add_jit_build_smoke "
                   "COMPILER_EXECUTABLE COMPILER_ENTRY\n";
      return 2;
    }
    if (!poisoned_helper_probe) {
      run_poisoned_helper_probe(argv);
    } else {
      preload_wrong_gen_ssig(argv[2]);
    }
    if (setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error("cannot select libtriton_jit engine");
    }

    TemporaryCache cache;
    flagdnn::Handle handle("hygon", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());

    fe::graph::Graph graph;
    graph.set_name("hygon_add_jit_build_smoke")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(true);
    const auto left = make_tensor(graph, "left", 1);
    const auto right = make_tensor(graph, "right", 2);
    auto output =
        graph.pointwise(left, right,
                        fe::graph::Pointwise_attributes()
                            .set_name("add")
                            .set_mode(fe::PointwiseMode_t::ADD)
                            .set_compute_data_type(fe::DataType_t::FLOAT)
                            .set_alpha(1.0));
    output->set_name("output")
        .set_uid(3)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1024})
        .set_stride({1})
        .set_output(true);

    const fe::error_t build_status = graph.build(handle, {fe::HeurMode_t::A});
    if (poisoned_helper_probe) {
      if (!build_status.is_bad() ||
          build_status.get_message().find(
              "loaded libtriton_jit Python helper differs from the CMake "
              "selection: gen_ssig") == std::string::npos) {
        throw std::runtime_error("poisoned gen_ssig was not rejected by the "
                                 "pre-use identity gate: " +
                                 build_status.get_message());
      }
      std::cout << "PASS preloaded wrong gen_ssig rejected before JIT use\n";
      return 0;
    }
    check_frontend(build_status, "FlagDNN Hygon Add JIT graph build");

    std::size_t manifest_count = 0;
    std::size_t selection_count = 0;
    std::size_t generated_source_count = 0;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(cache.path())) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const std::string filename = entry.path().filename().string();
      if (filename == "manifest.json") {
        ++manifest_count;
      }
      if (filename.starts_with(".flagdnn-autotune-v1-stage-")) {
        ++selection_count;
      }
      if (filename.starts_with("generated_stage_") &&
          entry.path().extension() == ".py") {
        ++generated_source_count;
      }
    }
    if (manifest_count != 1 || selection_count != 1 ||
        generated_source_count != 1) {
      throw std::runtime_error(
          "Hygon Add JIT build did not produce one manifest, one autotune "
          "selection, and one generated Triton source");
    }

    std::cout << "PASS Hygon Add C++ frontend -> JIT -> autotune build chain; "
              << "workspace=" << graph.get_workspace_size() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
