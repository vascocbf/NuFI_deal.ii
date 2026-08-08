#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

using LuaValue = std::variant<int, float, bool, std::string>;

class LuaConfig {
public:
  LuaConfig(const std::string &filePath, const std::string &tableName) {
    L = luaL_newstate();
    luaL_openlibs(L);

    if (luaL_dofile(L, filePath.c_str()) != LUA_OK) {
      std::string err = lua_tostring(L, -1);
      lua_close(L);
      throw std::runtime_error("Error loading Lua file: " + err);
    }

    lua_getglobal(L, tableName.c_str());
    if (!lua_istable(L, -1)) {
      lua_close(L);
      throw std::runtime_error("'" + tableName +
                               "' is not a table (or missing)");
    }
  }

  ~LuaConfig() {
    if (L)
      lua_close(L);
  }

  LuaConfig(const LuaConfig &) = delete;
  LuaConfig &operator=(const LuaConfig &) = delete;

  template <typename T> T get(const std::string &key) const {
    lua_getfield(L, -1, key.c_str());
    if (lua_isnil(L, -1)) {
      lua_pop(L, 1);
      throw std::runtime_error("Missing key: " + key);
    }
    T value = extract<T>(key);
    lua_pop(L, 1);
    std::cout << "[LuaConfig] " << key << " = " << value << "\n";
    return value;
  }

private:
  lua_State *L = nullptr;

  template <typename T> T extract(const std::string &key) const {
    if constexpr (std::is_same_v<T, int>) {
      if (!lua_isinteger(L, -1) && !lua_isnumber(L, -1))
        throw std::runtime_error("Key '" + key + "' is not a number");
      return static_cast<int>(lua_tointeger(L, -1));
    } else if constexpr (std::is_same_v<T, float>) {
      if (!lua_isnumber(L, -1))
        throw std::runtime_error("Key '" + key + "' is not a number");
      return static_cast<float>(lua_tonumber(L, -1));
    } else if constexpr (std::is_same_v<T, double>) {
      if (!lua_isnumber(L, -1))
        throw std::runtime_error("Key '" + key + "' is not a number");
      return static_cast<double>(lua_tonumber(L, -1));
    } else if constexpr (std::is_same_v<T, bool>) {
      if (!lua_isboolean(L, -1))
        throw std::runtime_error("Key '" + key + "' is not a boolean");
      return lua_toboolean(L, -1) != 0;
    } else if constexpr (std::is_same_v<T, std::string>) {
      if (!lua_isstring(L, -1))
        throw std::runtime_error("Key '" + key + "' is not a string");
      return std::string(lua_tostring(L, -1));
    } else {
      static_assert(!sizeof(T *), "Unsupported type for LuaConfig::get<T>()");
    }
  }
};
