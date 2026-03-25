from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout


class SentinelProbeAgentConan(ConanFile):
    name = "sentinel-probe-agent"
    version = "0.1.0"
    license = "Proprietary"
    description = "SENTINEL Probe Agent — GPU SDC detection daemon"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_tests": [True, False],
        "with_rocm": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_tests": True,
        "with_rocm": False,
    }

    def requirements(self):
        self.requires("grpc/1.60.0")
        self.requires("protobuf/25.1")
        self.requires("spdlog/1.13.0")
        self.requires("prometheus-cpp/1.2.4")
        self.requires("openssl/3.2.0")
        self.requires("nlohmann_json/3.11.3")
        self.requires("abseil/20230802.1", override=True)

    def build_requirements(self):
        if self.options.with_tests:
            self.test_requires("gtest/1.14.0")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["SENTINEL_BUILD_TESTS"] = self.options.with_tests
        tc.variables["SENTINEL_ENABLE_ROCM"] = self.options.with_rocm
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["sentinel_agent_lib", "sentinel_probes"]
