/// Implements a generic compile-time template-based pattern matching driver.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <cassert>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

namespace match {

namespace detail {

//===----------------------------------------------------------------------===//
// pattern_traits
//===----------------------------------------------------------------------===//

/// Trait class for analyzing invocable patterns.
///
/// The following members will be provided if @p T is a pattern:
///     - using result_type
///     - using signature = std::tuple<...>
///     - static constexpr bool is_closure
///     - static constexpr bool is_stateful
template<class T>
struct pattern_traits : std::false_type {};
// Accepts function pointers.
template<class Return, class... Matchers>
struct pattern_traits<Return (*)(Matchers...)> : std::true_type {
    using result_type = Return;
    using signature = std::tuple<Matchers...>;
    static constexpr bool is_closure = false;
    static constexpr bool is_stateful = false;
};
// Accepts pure closures.
template<class Return, class ClassType, class... Matchers>
struct pattern_traits<Return (ClassType::*)(Matchers...) const>
        : pattern_traits<Return (*)(Matchers...)> {
    static constexpr bool is_closure = true;
};
// Accepts impure closures.
template<class Return, class ClassType, class... Args>
struct pattern_traits<Return (ClassType::*)(Args...)>
        : pattern_traits<Return (ClassType::*)(Args...) const> {
    static constexpr bool is_stateful = true;
};
// Accepts closures.
template<class T>
    requires (std::is_class_v<T>)
struct pattern_traits<T> : pattern_traits<decltype(&T::operator())> {};

/// Concept for invocable patterns.
template<class T>
concept Pattern = pattern_traits<std::decay_t<T>>::value;

} // namespace detail

//===----------------------------------------------------------------------===//
// Match<T>
//===----------------------------------------------------------------------===//

namespace detail {

template<class Context>
struct MatchAssembler;

} // namespace detail

/// Container that stores the result of a matcher.
///
/// A match is either empty / failed or contains a value of type @p T . The type
/// can be anything except void, including (rvalue) references.
///
/// @pre        @p T is not void.
///
/// @tparam     T   Result binding type.
template<class T>
    requires (!std::is_void_v<T>)
class [[nodiscard]] Match : std::optional<std::tuple<T>> {
    using Impl = std::optional<std::tuple<T>>;

    [[nodiscard]] constexpr std::tuple<T> &&unwrap() &&
    {
        assert(has_value());
        return std::move(static_cast<Impl &>(*this)).value();
    }

    template<class Context>
    friend struct detail::MatchAssembler;

public:
    /// Initializes a failed, empty Match.
    ///
    /// @post   `!has_value()`
    /*implicit*/ constexpr Match() : Impl() { assert(!has_value()); }
    /// @copydoc Match()
    ///
    /// @post   `!has_value()`
    /*implicit*/ constexpr Match(std::nullopt_t) : Impl()
    {
        assert(!has_value());
    }
    /// Initializes a successful Match by constructing @p T from @p args .
    ///
    /// @pre    @p T is constructible from @p Args .
    /// @post   `has_value()`
    template<class... Args>
        requires (std::is_constructible_v<T, Args...>)
    /*implicit*/ constexpr Match(std::in_place_t, Args &&...args)
            : Impl(std::in_place, T(std::forward<decltype(args)>(args)...))
    {
        assert(has_value());
    }
    /// Initializes a successful Match by constructing @p T from @p u .
    ///
    /// @pre    @p T is constructible from @p u .
    /// @post   `has_value()`
    template<class U>
        requires (std::is_constructible_v<T, U>)
    /*implicit*/ constexpr Match(U &&u)
            : Match(std::in_place, std::forward<U>(u))
    {
        assert(has_value());
    }
    /// Initializes a match by trying to unwrap @p opt .
    ///
    /// @post   `has_value() == opt_before.has_value()`
    /*implicit*/ constexpr Match(std::optional<T> &&opt)
        requires (!std::is_reference_v<T>)
            : Match()
    {
        if (opt) {
            emplace(std::move(opt).value());
            assert(has_value());
        } else {
            assert(!has_value());
        }
    }
    /// Initializes a match by trying to unwrap @p opt .
    ///
    /// @post   `has_value() == opt.has_value()`
    /*implicit*/ constexpr Match(const std::optional<T> &opt)
        requires (!std::is_reference_v<T>)
            : Match()
    {
        if (opt) emplace(opt.value());
        assert(has_value() == opt.has_value());
    }

    /// Gets a value indicating whether this Match is successful.
    [[nodiscard]] constexpr bool has_value() const { return Impl::has_value(); }
    /// @copydoc has_value()
    /*implicit*/ constexpr operator bool() const { return has_value(); }

    /// Unwraps the contained successful Match value.
    ///
    /// @pre    `has_value()`
    [[nodiscard]] constexpr decltype(auto) value() const &
    {
        return std::get<0>(Impl::value());
    }
    /// @copydoc value()
    [[nodiscard]] constexpr decltype(auto) operator*() const &
    {
        return value();
    }
    /// @copydoc value()
    [[nodiscard]] constexpr decltype(auto) value() &&
    {
        return std::get<0>(Impl::value());
    }
    /// @copydoc value()
    [[nodiscard]] constexpr decltype(auto) operator*() const &&
    {
        return value();
    }

    /// Swaps the contents of two Match instances.
    constexpr friend void swap(Match &lhs, Match &rhs)
    {
        swap(static_cast<Impl &>(lhs), static_cast<Impl &>(rhs));
    }
    /// Resets this Match to the failed, empty state.
    ///
    /// @post   `!has_value()`
    constexpr void reset() { Impl::reset(); }
    /// Overwrites this Match by constructing @p T from @p args.
    ///
    /// @post   `has_value()`
    constexpr T &emplace(auto &&...args)
    {
        return std::get<0>(
            Impl::emplace(T(std::forward<decltype(args)>(args)...)));
    }
};

/// Result value that indicates a match failure.
///
/// Guarantees that `!Match<T>(match_fail).has_value()`.
static constexpr auto match_fail = std::nullopt;

//===----------------------------------------------------------------------===//
// ContextBase
//===----------------------------------------------------------------------===//

/// Container that stores the result of a pattern match.
///
/// Wraps @p T in a Match<T>, except when @p T is void, in which case the
/// result is an std::optional of an std::monostate (i.e. fallible but empty).
///
/// @tparam     T   Result type.
template<class T>
using PatternResult = std::
    conditional_t<std::is_void_v<T>, std::optional<std::monostate>, Match<T>>;

/// Base class for a pattern matching context returning @p Result .
///
/// @tparam     Result      Pattern matching result type.
template<class Result = void>
struct ContextBase {
    /// The type of the pattern matching result.
    using result_type = Result;

    /// Initializes the initial matching state.
    ///
    /// The default implementation creates an std::monostate instance.
    [[nodiscard]] constexpr std::monostate init_state() const { return {}; }

    /// Matches the next @p Matcher using @p state .
    ///
    /// Must return an `std::pair<Match<Matcher>, NewState>` instance. If the
    /// match is contextually convertible to @c false , the pattern will be
    /// rejected.
    template<class Matcher>
    constexpr auto match(auto &&state) const;

    /// Finalizes the assembly of a pattern in @p state .
    ///
    /// If the result is @c false , the assembled pattern will be rejected.
    ///
    /// The default implementation accepts every assembled pattern.
    [[nodiscard]] constexpr bool assemble(auto &&state) const
    {
        std::ignore = state;
        return true;
    }

    /// Accepts a @p pattern_result to attempt completing the pattern match.
    ///
    /// If the result is contextually convertible to @c false , the pattern
    /// application will be rejected.
    [[nodiscard]] constexpr PatternResult<result_type>
    complete(auto &&state, auto &&pattern_result) const;
};

/// Base class for implementing stateless pattern matching contexts.
///
/// @tparam     Derived     CRTP derived class.
/// @tparam     Result      Pattern matching result type.
template<class Derived, class Result>
class StatelessContext : public ContextBase<Result> {
    const Derived &_self() const { return static_cast<const Derived &>(*this); }

public:
    /// @copydoc ContextBase::result_type
    using result_type = typename ContextBase<Result>::result_type;

    /// Statelss match function.
    template<class Matcher>
    constexpr Match<Matcher> match_stateless() const;
    /// Delegates to the stateless match function.
    template<class Matcher>
    constexpr std::pair<Match<Matcher>, std::monostate>
    match(std::monostate) const
    {
        if (auto match = _self().template match_stateless<Matcher>())
            return std::make_pair(std::move(match), std::monostate{});

        return {};
    }

    /// Stateless assembly function.
    ///
    /// The default implementation accepts every assembled pattern.
    [[nodiscard]] constexpr bool assemble_stateless() const { return true; }
    /// Delegates to the stateless assembly function.
    [[nodiscard]] constexpr bool assemble(std::monostate) const
    {
        return _self().assemble_stateless();
    }

    /// Stateless completion function.
    [[nodiscard]] constexpr PatternResult<result_type>
    complete_stateless(auto &&pattern_result) const;
    /// Forwards a matching result value.
    [[nodiscard]] constexpr PatternResult<result_type>
    complete_stateless(result_type pattern_result) const
    {
        return std::forward<result_type>(pattern_result);
    }
    /// Delegates to the stateless completion function.
    [[nodiscard]] constexpr PatternResult<result_type>
    complete(std::monostate, auto &&pattern_result) const
    {
        return _self().complete_stateless(
            std::forward<decltype(pattern_result)>(pattern_result));
    }
};

namespace detail {

//===----------------------------------------------------------------------===//
// MatchAssembler
//===----------------------------------------------------------------------===//

template<std::size_t, class>
struct tuple_skip;
template<>
struct tuple_skip<0, std::tuple<>> : std::type_identity<std::tuple<>> {};
template<std::size_t I, class Head, class... Tail>
struct tuple_skip<I, std::tuple<Head, Tail...>>
        : std::conditional_t<
              (I > 0),
              tuple_skip<(I - 1), std::tuple<Tail...>>,
              std::type_identity<std::tuple<Head, Tail...>>> {};
template<std::size_t I, class Tuple>
using tuple_skip_t = typename tuple_skip<I, Tuple>::type;

/// Assembles matchers for a pattern in @p Context .
template<class Context>
struct MatchAssembler {
    /// Assembles matchers recursively.
    ///
    /// Attempts to match the head matcher, and then the tail matchers. Fails
    /// when the first matcher fails. Otherwise, the match result is assembled
    /// into a tuple.
    ///
    /// @pre    @p Signature is a tuple of matcher types.
    /// @pre    `MatcherIdx <= std::tuple_size_v<Signature>`
    template<class Signature, std::size_t MatcherIdx = 0>
    [[nodiscard]] static constexpr auto assemble()
    {
        if constexpr (MatcherIdx == std::tuple_size_v<Signature>) {
            return []<class State>(const Context &ctx, State state) constexpr
                   -> std::optional<std::tuple<State>> {
                // Call `ctx.assemble` to determine whether the pattern is
                // accepted.
                if (!ctx.assemble(std::forward<decltype(state)>(state)))
                    return match_fail;
                return std::optional<std::tuple<State>>(
                    std::in_place,
                    std::move(state));
            };
        } else {
            using head_matcher = std::tuple_element_t<MatcherIdx, Signature>;
            constexpr auto tail_matchers =
                assemble<Signature, MatcherIdx + 1>();

            return [=](const Context &ctx, auto &&state) constexpr -> auto {
                // Attempt to match the Head matcher.
                auto [head, new_state] = match<head_matcher>(
                    ctx,
                    std::forward<decltype(state)>(state));

                using result_tuple = decltype(std::tuple_cat(
                    std::move(head).unwrap(),
                    std::move(*tail_matchers(ctx, std::move(new_state)))));
                using result_type = std::optional<result_tuple>;

                // If the head matcher failed, we stop immediately.
                if (!head) return result_type(match_fail);
                // Attempt to match the Tail matchers.
                auto tail = tail_matchers(ctx, std::move(new_state));
                // If any of the tail matchers failed, we stop immediately.
                if (!tail) return result_type(match_fail);
                // Assemble the result tuple.
                return result_type(
                    std::in_place,
                    std::tuple_cat(std::move(head).unwrap(), std::move(*tail)));
            };
        }
    }

private:
    /// Performs a single match.
    ///
    /// This wrapper ensures that patterns can always match the @p Context via
    /// implicit conversions.
    ///
    /// @pre    @p sate is a valid state for @p Context .
    template<class Matcher>
    [[nodiscard]] static constexpr auto match(const Context &ctx, auto &&state)
    {
        if constexpr (std::is_convertible_v<const Context &, Matcher>) {
            return std::make_pair(
                Match<Matcher>(static_cast<Matcher>(ctx)),
                std::forward<decltype(state)>(state));
        } else {
            return ctx.template match<Matcher>(
                std::forward<decltype(state)>(state));
        }
    }
};

} // namespace detail

//===----------------------------------------------------------------------===//
// AssembledPattern
//===----------------------------------------------------------------------===//

namespace detail {

/// Applies @p tuple to @p fn , except for the last index.
constexpr decltype(auto) apply_minus_one(auto &&fn, auto &&tuple)
{
    constexpr auto impl = []<std::size_t... Is>(
                              std::index_sequence<Is...>,
                              auto &&fn,
                              auto &&tuple) constexpr -> decltype(auto) {
        return fn(std::get<Is>(std::forward<decltype(tuple)>(tuple))...);
    };

    constexpr auto arity = std::tuple_size_v<std::decay_t<decltype(tuple)>>;
    static_assert(arity >= 1, "Tuple is too small");

    return impl(
        std::make_index_sequence<arity - 1>{},
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(tuple)>(tuple));
}

} // namespace detail

/// Functor that wraps an assembled @p Pattern for @p Context .
///
/// @pre    @p Context is a pattern matching context.
/// @pre    @p Pattern is an invocable pattern.
template<class Context, detail::Pattern Pattern>
class AssembledPattern {
    using pattern_traits = detail::pattern_traits<Pattern>;
    using signature = typename pattern_traits::signature;
    static constexpr auto arity = std::tuple_size_v<signature>;
    static constexpr auto matcher =
        detail::MatchAssembler<Context>::template assemble<signature>();

public:
    /// The underlying pattern type.
    using pattern_type = Pattern;
    /// The type of the pattern result.
    using result_type = PatternResult<typename Context::result_type>;

    /// Initializes an AssembledPattern from @p pattern .
    /*implicit*/ constexpr AssembledPattern(const Pattern &pattern)
            : m_pattern(pattern)
    {}
    /// @copydoc AssembledPattern(const Pattern &).
    /*implicit*/ constexpr AssembledPattern(Pattern &&pattern)
            : m_pattern(std::move(pattern))
    {}

    /// Invokes the pattern on @p ctx .
    [[nodiscard]] constexpr result_type operator()(const Context &ctx) const
        requires (!pattern_traits::is_stateful)
    {
        return invoke(ctx, m_pattern);
    };
    /// @copydoc operator()(const Context &) const
    constexpr result_type operator()(const Context &ctx)
    {
        return invoke(ctx, m_pattern);
    };

private:
    [[nodiscard]] static result_type invoke(const Context &ctx, auto &pattern)
    {
        // Assemble and invoke the matchers.
        auto match = matcher(ctx, ctx.init_state());
        // If any of the matchers has failed, we stop immediately.
        if (!match) return match_fail;
        // Apply the pattern and the completer.
        return ctx.complete(
            std::get<arity>(std::move(*match)),
            detail::apply_minus_one(pattern, std::move(*match)));
    }

    [[no_unique_address]] Pattern m_pattern;
};

/// Assembles @p pattern for @p Context .
///
/// @pre    @p Context is a pattern matching context.
template<class Context>
[[nodiscard]] constexpr auto make_pattern(detail::Pattern auto &&pattern)
{
    return AssembledPattern<Context, std::decay_t<decltype(pattern)>>(
        std::forward<decltype(pattern)>(pattern));
}

//===----------------------------------------------------------------------===//
// PatternMatch
//===----------------------------------------------------------------------===//

namespace detail {

/// Trait that determines whether something is an assembled pattern.
template<class>
struct is_assembled_pattern : std::false_type {};
template<class Context, class Pattern>
struct is_assembled_pattern<AssembledPattern<Context, Pattern>>
        : std::true_type {};
template<class T>
static constexpr auto is_assembled_pattern_v = is_assembled_pattern<T>::value;

} // namespace detail

/// Performs a pattern match for @p Context .
///
/// @pre    @p Context is a pattern matching context.
template<class Context>
class PatternMatch : PatternResult<typename Context::result_type> {
    using Impl = PatternResult<typename Context::result_type>;

    [[nodiscard]] constexpr Impl &storage()
    {
        return static_cast<Impl &>(*this);
    }
    [[nodiscard]] constexpr const Impl &storage() const
    {
        return static_cast<const Impl &>(*this);
    }

public:
    /// The type of the pattern matching result.
    using result_type = typename Context::result_type;
    /// Indicates whether this PatternMatch has a result value.
    static constexpr auto has_result = !std::is_void_v<result_type>;

    /// Initializes a PatternMatch for @p ctx .
    /*implicit*/ constexpr PatternMatch(const Context &ctx) : Impl(), m_ctx(ctx)
    {}

    /*implicit*/ PatternMatch(const PatternMatch &) = delete;
    /*implicit*/ PatternMatch(PatternMatch &&) = delete;
    PatternMatch &operator=(const PatternMatch &) = delete;
    PatternMatch &operator=(PatternMatch &&) = delete;

    /// Gets the underlying context.
    [[nodiscard]] constexpr const Context &context() const { return m_ctx; }

    /// Gets a value indicating whether the match succeeded.
    [[nodiscard]] constexpr bool has_match() const
    {
        return storage().has_value();
    }
    /// @copydoc has_match()
    /*implicit*/ constexpr operator bool() const { return has_match(); }

    /// Attempts to match @p pattern .
    ///
    /// @post   `!before.has_match() || has_match()`
    constexpr PatternMatch &match(detail::Pattern auto &&pattern)
    {
        if constexpr (detail::is_assembled_pattern_v<
                          std::decay_t<decltype(pattern)>>) {
            if (has_match()) return *this;
            storage() = pattern(context());
            return *this;
        } else {
            return match(make_pattern<Context>(
                std::forward<decltype(pattern)>(pattern)));
        }
    }
    /// Attempts to match @p Matcher , and emplace a result using @p args .
    ///
    /// @post   `!before.has_match() || has_match()`
    template<class Matcher, class... Args>
        requires (std::is_constructible_v<result_type, Args...>)
    constexpr PatternMatch &match(Args &&...args)
    {
        if (has_match()) return *this;
        if (context().template match<Matcher>(context().init_state()).first)
            storage().emplace(std::forward<Args>(args)...);
        return *this;
    }

    /// Returns the match result, or constructs a default from @p args .
    ///
    /// @pre    `has_result`
    template<class... Args>
        requires (has_result && std::is_constructible_v<result_type, Args...>)
    constexpr result_type or_default(Args &&...args)
    {
        if (has_match()) return std::move(Impl::value());
        return result_type(std::forward<Args>(args)...);
    }
    /// Returns the match result, or constructs a default using @p fallback .
    template<class Fallback>
        requires (std::is_invocable_r_v<result_type, Fallback, const Context &>)
    constexpr result_type or_default(Fallback &&fallback)
    {
        if constexpr (has_result) {
            // Handle non-void result case.
            if (has_match()) return std::move(Impl::value());
            return fallback(context());
        } else {
            // Handle void result case.
            if (has_match()) return;
            fallback(context());
        }
    }

private:
    const Context m_ctx;
};

//===----------------------------------------------------------------------===//
// AssembledMatcher
//===----------------------------------------------------------------------===//

namespace detail {

/// Applies a tuple of @p patterns to @p ctx in sequence, returning first match.
///
/// @pre    @p patterns is a tuple of patterns.
/// @pre    @p ctx is a pattern matching context.
template<std::size_t I = 0>
constexpr auto apply(auto &patterns, const auto &ctx)
    -> PatternResult<typename std::decay_t<decltype(ctx)>::result_type>
{
    using pattern_tuple = std::decay_t<decltype(patterns)>;
    if constexpr (I >= std::tuple_size_v<pattern_tuple>) {
        return match_fail;
    } else {
        auto &pattern = std::get<I>(patterns);
        if (auto result = pattern(ctx)) return std::move(result);
        return apply<(I + 1)>(patterns, ctx);
    }
}

} // namespace detail

/// Stores pre-assembled @p Patterns for pattern matching on @p Context .
///
/// @pre    @p Context is a pattern matching context.
template<class Context, detail::Pattern... Patterns>
class AssembledMatcher {
public:
    /// The type of the pattern matching result.
    using result_type = typename Context::result_type;
    /// Indicates whether this PatternMatch has a result value.
    static constexpr auto has_result = !std::is_void_v<result_type>;

    /// The result type of an individual pattern.
    using pattern_result = PatternResult<typename Context::result_type>;
    /// The type that stores the contained patterns.
    using pattern_storage = std::tuple<AssembledPattern<Context, Patterns>...>;

    /*implicit*/ constexpr AssembledMatcher()
        requires (sizeof...(Patterns) == 0)
            : m_patterns()
    {}
    explicit constexpr AssembledMatcher(pattern_storage patterns)
            : m_patterns(std::move(patterns))
    {}

    /// Appends @p patterns to the end of the matcher.
    template<detail::Pattern... TailPatterns>
    constexpr auto match(AssembledMatcher<Context, TailPatterns...> patterns) &&
    {
        return AssembledMatcher<Context, Patterns..., TailPatterns...>(
            std::tuple_cat(
                std::move(m_patterns),
                std::move(patterns.m_patterns)));
    }
    /// Appends @p pattern to the end of the matcher.
    constexpr auto match(detail::Pattern auto &&pattern) &&
    {
        using Pattern = std::decay_t<decltype(pattern)>;
        if constexpr (detail::is_assembled_pattern_v<Pattern>) {
            using nested_pattern = typename Pattern::pattern_type;
            return AssembledMatcher<Context, Patterns..., nested_pattern>(
                std::tuple_cat(
                    std::move(m_patterns),
                    std::forward_as_tuple(pattern)));
        } else {
            return std::move(*this).match(make_pattern<Context>(
                std::forward<decltype(pattern)>(pattern)));
        }
    }
    /// Appends a match on @p Matcher constructing a result from @p args .
    template<class Matcher, class... Args>
        requires (std::is_constructible_v<result_type, Args...>)
    constexpr auto match(Args... args) &&
    {
        return std::move(*this).match(
            make_pattern<Context>([=](Matcher) constexpr -> result_type {
                return result_type(args...);
            }));
    }

    /// Appends a fallback matcher constructing a result from @p args .
    ///
    /// @pre    `has_result`
    template<class... Args>
        requires (has_result && std::is_constructible_v<result_type, Args...>)
    constexpr auto or_default(Args... args) &&
    {
        return std::move(*this).match(make_pattern<Context>(
            [=]() constexpr -> result_type { return result_type(args...); }));
    }
    /// Appends a @p fallback matcher.
    template<class Fallback>
        requires (std::is_invocable_r_v<result_type, Fallback, const Context &>)
    constexpr auto or_default(Fallback fallback) &&
    {
        // Note that this also works for !has_result.
        return std::move(*this).match(make_pattern<Context>(
            [fallback = std::move(fallback)](const Context &ctx) constexpr
            -> result_type { return fallback(ctx); }));
    }

    /// Obtains the stored patterns.
    constexpr const pattern_storage &patterns() const & { return m_patterns; }
    /// @copydoc patterns()
    constexpr pattern_storage &&patterns() && { return std::move(m_patterns); }

    /// Applies the matcher to @p ctx .
    constexpr pattern_result operator()(const Context &ctx)
    {
        return detail::apply(m_patterns, ctx);
    }
    /// @copydoc operator()(const Context &)
    constexpr pattern_result operator()(const Context &ctx) const
    {
        return detail::apply(m_patterns, ctx);
    }

private:
    [[no_unique_address]] pattern_storage m_patterns;
};

//===----------------------------------------------------------------------===//
// match
//===----------------------------------------------------------------------===//

/// Trait class for adding implied pattern matching contexts to types.
///
/// Specializations should derive std::true_type to indicate that @p T has an
/// implied context.
template<class T>
struct ImpliedContext : std::false_type {};

/// Concept for a type that has an implied pattern matching context.
template<class T>
concept Matchable = ImpliedContext<T>::value;
/// Concept for a matchable type that has an implicit result type.
template<class T>
concept DefaultMatchable =
    Matchable<T> && requires { typename ImpliedContext<T>::result_type; };
/// Concept for a matchable type that accepts explicit result types.
template<class T, class Result>
concept MatchableAs = Matchable<T> && requires {
    typename ImpliedContext<T>::template bind<Result>;
};

/// Starts a pattern match for @p matchable .
[[nodiscard]] constexpr auto match(DefaultMatchable auto &matchable)
{
    using context = ImpliedContext<std::decay_t<decltype(matchable)>>;
    return PatternMatch<context>(matchable);
}
/// Starts an explicitly-typed pattern match for @p matchable .
template<class Result>
[[nodiscard]] constexpr auto match(MatchableAs<Result> auto &matchable)
{
    using base = ImpliedContext<std::decay_t<decltype(matchable)>>;
    using context = typename base::template bind<Result>;
    return PatternMatch<context>(matchable);
}

//===----------------------------------------------------------------------===//
// matcher
//===----------------------------------------------------------------------===//

/// Starts an AssembledMatcher for @p Matchable with the default result type.
template<DefaultMatchable Matchable>
[[nodiscard]] constexpr auto default_matcher()
{
    using context = ImpliedContext<Matchable>;
    return AssembledMatcher<context>();
}

/// Starts an AssembledMatcher for @p Matchable yielding @p Result .
template<class Result, MatchableAs<Result> Matchable>
[[nodiscard]] constexpr auto matcher()
{
    using base = ImpliedContext<Matchable>;
    using context = typename base::template bind<Result>;
    return AssembledMatcher<context>();
}

} // namespace match

//===----------------------------------------------------------------------===//
// VariantMatcher
//===----------------------------------------------------------------------===//

namespace match {

/// Implements a StatelessContext for std::variant .
template<class Result, class... Types>
class VariantMatcher
        : public match::
              StatelessContext<VariantMatcher<Result, Types...>, Result> {
    template<class T>
    static constexpr bool has_alternative_v =
        std::disjunction_v<std::is_same<T, Types>...>;

public:
    VariantMatcher(const std::variant<Types...> &variant) : m_variant(variant)
    {}

    template<class T>
    constexpr match::Match<T> match_stateless() const
    {
        if constexpr (has_alternative_v<T>) {
            if (std::holds_alternative<T>(m_variant))
                return std::get<T>(m_variant);
        }

        return match::match_fail;
    }

private:
    const std::variant<Types...> &m_variant;
};

template<class... Ts>
struct ImpliedContext<std::variant<Ts...>> : std::true_type {
    template<class Result>
    using bind = VariantMatcher<Result, Ts...>;
};

} // namespace match
