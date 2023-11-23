/// Implements the constant folding application template.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "ub-mlir/Dialect/UBX/IR/Attributes.h"

#include <type_traits>
#include <utility>

namespace mlir::ubx {

namespace detail {

template<class Fn, class T>
struct apply_accu {
    /*implicit*/ apply_accu(Fn fn) : m_fn(std::move(fn)), m_result() {}

    void reserve(std::size_t count) { m_result.reserve(count); }

    void operator()(auto &&...args)
    {
        m_result.emplace_back(m_fn(std::forward<decltype(args)>(args)...));
    }

    SmallVector<T> unwrap() && { return std::move(m_result); }

private:
    [[no_unique_address]] Fn m_fn;
    SmallVector<T> m_result;
};

template<class Fn, class T>
struct apply_accu<Fn, std::optional<T>> {
    /*implicit*/ apply_accu(Fn fn) : m_fn(std::move(fn)), m_values(), m_mask()
    {}

    void reserve(std::size_t count)
    {
        m_values.reserve(count);
        m_mask.reserve(count);
    }

    void operator()(auto &&...args)
    {
        if (auto element = m_fn(std::forward<decltype(args)>(args)...)) {
            m_values.emplace_back(std::move(element).value());
            m_mask.push_back(false);
            return;
        }

        m_values.emplace_back();
        m_mask.push_back(true);
    }

    std::pair<SmallVector<T>, SmallVector<bool>> unwrap() &&
    {
        return std::make_pair(std::move(m_values), std::move(m_mask));
    }

private:
    [[no_unique_address]] Fn m_fn;
    SmallVector<T> m_values;
    SmallVector<bool> m_mask;
};

template<class Fn>
struct apply_accu<Fn, void> {
    /*implicit*/ apply_accu(Fn fn) : m_fn(std::move(fn)) {}

    void reserve(std::size_t) {}

    void operator()(auto &&...args)
    {
        m_fn(std::forward<decltype(args)>(args)...);
    }

    void unwrap() && { return; }

private:
    [[no_unique_address]] Fn m_fn;
};

template<class Fn, class... Operands>
using apply_accu_t =
    apply_accu<Fn, std::invoke_result_t<Fn, typename Operands::FoldType...>>;

} // namespace detail

/// Applies @p fn to all values contained in @p operands .
///
/// The result type is determined by the result type of @p fn :
///
///     - If @p fn returns void, apply also returns void.
///     - If @p fn returns an std::optional<T>, apply returns a pair of
///       SmallVector instances, containing T and bool respectively.
///     - If @p fn returns anything else, apply returns a SmallVector of it.
///
/// @pre    There must be at least one operand.
/// @pre    @p operands must be derived from ValueOrPoisonLikeAttr.
/// @pre    @p fn must be invocable on the fold types of @p operands .
/// @pre    The shapes of all @p operands must match.
template<class Fn, class... Operands>
    requires(
        (std::is_base_of_v<detail::ValueOrPoisonLikeAttrBase, Operands> && ...)
        && std::is_invocable_v<Fn, typename Operands::FoldType...>
        && sizeof...(Operands) > 0)
inline auto apply(Fn fn, Operands... operands)
{
    detail::apply_accu_t<Fn, Operands...> accu(std::move(fn));
    std::tuple its((operands.value_begin())...);
    const auto fwd = [&](auto... args) { accu((*args)...); };
    const auto inc = [&](auto &...args) { (++args, ...); };

    std::tuple ops(operands...);
    if (std::apply(
            [](auto... args) { return ((args.isSplat()) && ...); },
            ops)) {
        // Splat case.
        std::apply(fwd, its);
    } else {
        // Dense case.
        const auto size = std::get<0>(ops).size();
        const auto all_same_size = [=](auto... args) {
            return ((args.size() == size) && ...);
        };
        assert(std::apply(all_same_size, ops));

        accu.reserve(size);
        for (std::size_t i = 0; i < size; ++i) {
            std::apply(fwd, its);
            std::apply(inc, its);
        }
    }

    return std::move(accu).unwrap();
}

} // namespace mlir::ubx
