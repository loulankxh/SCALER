#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <iomanip>
#include <fmt/core.h>
#include <fmt/chrono.h>

namespace scalegann {

enum class LogLevel {
    INFO,
    DEBUG,
    WARN,
    ERR,
    SUCCESS,
    TASK,
    SPOT
};

class Logger {
public:
    template<typename... Args>
    static void log(LogLevel level, int id, std::string_view format, Args&&... args) {
        std::lock_guard<std::mutex> lock(getMutex());
        
        auto now = std::chrono::system_clock::now();
        std::string time_str = fmt::format("{:%H:%M:%S}", now);
        
        std::string prefix;
        switch (level) {
            case LogLevel::INFO:    prefix = "\033[34m[INFO]\033[0m"; break;
            case LogLevel::DEBUG:   prefix = "\033[36m[DEBUG]\033[0m"; break;
            case LogLevel::WARN:    prefix = "\033[33m[WARN]\033[0m"; break;
            case LogLevel::ERR:     prefix = "\033[31m[ERROR]\033[0m"; break;
            case LogLevel::SUCCESS: prefix = "\033[32m[SUCCESS]\033[0m"; break;
            case LogLevel::TASK:    prefix = "\033[35m[TASK]\033[0m"; break;
            case LogLevel::SPOT:    prefix = "\033[33;1m[SPOT]\033[0m"; break;
        }

        std::string id_str = (id != -1) ? fmt::format(" \033[1m[ID:{}]\033[0m", id) : "";
        
        std::string message = fmt::format(fmt::runtime(format), std::forward<Args>(args)...);
        
        fmt::print("[{}] {} {}{}\n", time_str, prefix, id_str, message);
    }

    template<typename... Args>
    static void info(std::string_view format, Args&&... args) { log(LogLevel::INFO, -1, format, args...); }
    
    template<typename... Args>
    static void success(std::string_view format, Args&&... args) { log(LogLevel::SUCCESS, -1, format, args...); }
    
    template<typename... Args>
    static void error(std::string_view format, Args&&... args) { log(LogLevel::ERR, -1, format, args...); }

    template<typename... Args>
    static void task(int id, std::string_view format, Args&&... args) { log(LogLevel::TASK, id, format, args...); }
    
    template<typename... Args>
    static void spot(int id, std::string_view format, Args&&... args) { log(LogLevel::SPOT, id, format, args...); }

private:
    static std::mutex& getMutex() {
        static std::mutex mtx;
        return mtx;
    }
};

} // namespace scalegann
