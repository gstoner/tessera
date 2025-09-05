
The C++ microbenchmarks expect the single-header **nlohmann/json** to be available in your include path.

On Debian/Ubuntu:
```bash
sudo apt-get install nlohmann-json3-dev
```

Or vendored:
```bash
wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -O third_party/json.hpp
# then add -Ithird_party when building, or place the header system-wide
```
