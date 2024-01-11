#ifndef KOUEK_META_TYPE_H
#define KOUEK_META_TYPE_H

namespace kouek {

struct Noncopyable {
    Noncopyable() = default;
    Noncopyable(const Noncopyable &) = delete;
    Noncopyable &operator=(const Noncopyable &) = delete;
};

template <typename T> struct Modifiable {
    using DataType = T;

    bool modified = true;
    T dat;

    template <typename U> void Set(U T::*member, const U &val) {
        if (dat.*member != val) {
            dat.*member = val;
            modified = true;
        }
    }
    const T &GetAndReset() {
        modified = false;
        return dat;
    }
};

} // namespace kouek

#endif // !KOUEK_META_TYPE_H
