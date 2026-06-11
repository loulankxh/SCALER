#include <atomic>
#include <type_traits>

template <typename T>
struct AtomicWrapper {
    static_assert(std::is_arithmetic<T>::value, "AtomicWrapper<T> must be numeric value!!!");

    std::atomic<T> _a;

    AtomicWrapper() :_a(0) {}

    AtomicWrapper(const T &a) :_a(std::atomic<T>(a).load()) {}

    AtomicWrapper(const std::atomic<T> &a) :_a(a.load()) {}

    AtomicWrapper(const AtomicWrapper &other) :_a(other._a.load()) {}

    AtomicWrapper &operator=(const AtomicWrapper &other) {
        _a.store(other._a.load());
        return *this;
    }

    AtomicWrapper &operator=(const std::atomic<T> &a) {
        _a.store(a.load());
        return *this;
    }

    AtomicWrapper &operator=(const T &a) {
        _a.store(a);
        return *this;
    }

    T load() {
        return _a.load();
    }

    T load_relaxed() {
        return _a.load(std::memory_order_relaxed);
    }

    void store(const std::atomic<T> &a) {
        _a.store(a);
    }

    T operator++(int) {
        return _a.fetch_add(1);
    }

    T fetch_add_relaxed(T v) {
        return _a.fetch_add(v, std::memory_order_relaxed);
    }
};