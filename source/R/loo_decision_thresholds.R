#' Cost-optimal two-threshold decisions from PSIS-LOO risk probabilities
#'
#' `loo_decision_thresholds()` turns leave-one-out risk probabilities into a
#' reusable three-way decision rule. Each unit (for example a transaction) is
#' assigned to one of three bands: **approve**, **manual review**, or **block**.
#' On a labelled batch the rule is chosen to minimize the total cost subject to
#' an upper bound on the number of reviews and an upper bound on the number of
#' false blocks.
#'
#' The risk probability of unit \eqn{i} is the leave-one-out predictive mean of
#' the probability that \eqn{y_i = 1}, computed with [E_loo()] from the PSIS
#' approximation returned by [psis()]. Passing a matrix of posterior draws of
#' the event probability therefore gives the same quantity that
#' [loo_predictive_metric()] uses for binary classification, while leaving the
#' existing accuracy metrics untouched.
#'
#' @details
#' ## Cost model
#'
#' Let \eqn{y_i = 1} denote the adverse outcome (for example a chargeback) and
#' \eqn{y_i = 0} the benign outcome. A decision costs
#' \describe{
#'   \item{approve}{`cost_false_approve` if \eqn{y_i = 1} (a false approve),
#'     and nothing if \eqn{y_i = 0}.}
#'   \item{block}{`cost_false_block` if \eqn{y_i = 0} (a false block), and
#'     nothing if \eqn{y_i = 1}.}
#'   \item{review}{`cost_review`, regardless of \eqn{y_i}.}
#' }
#' A reviewed unit is assumed to be decided correctly, so it never incurs a
#' false-approve or false-block cost. The objective is the total cost over the
#' labelled batch. Because every reviewed unit costs the same and is never
#' misclassified, the optimal rule is a function of the risk alone: approve the
#' lowest risks, review the middle, and block the highest risks.
#'
#' ## Threshold rule and ties
#'
#' The returned `thresholds` are the cut points of the half-open rule
#' \preformatted{
#'   approve if risk <  low
#'   review  if low <= risk < high
#'   block   if risk >= high
#' }
#' so the rule can be applied unchanged to new units. `low` is the smallest
#' reviewed risk and `high` is the smallest blocked risk. Any of the three
#' bands may be empty: with no reviewed units `low >= high`, an empty approve
#' band gives `low = min(risk)`, and an empty block band gives `high = Inf`.
#' When the whole batch is approved both thresholds are `Inf`, and when the
#' whole batch is blocked both equal `min(risk)`.
#'
#' Units with an identical risk probability are never split across bands: cut
#' points are always placed between distinct risk values. Identical means
#' exactly equal in floating point (zero-width tolerance), so the rule is
#' deterministic and reproduces the same decisions when applied to the
#' labelled batch.
#'
#' ## Constraints
#'
#' `max_reviews` bounds the total number of reviewed units and
#' `max_false_block` bounds the number of blocked units with \eqn{y_i = 0}.
#' Both may be `Inf` (the default) and are non-binding when they exceed the
#' corresponding totals. If no rule can satisfy a bound the function errors;
#' this can only happen for `max_false_block` below the number of benign units
#' once all of them must be blocked, and even then approving everything is
#' always feasible for the defaults.
#'
#' ## Tie-breaking
#'
#' Among rules with the same total cost (up to a small relative tolerance),
#' the rule with fewer reviews is returned. Remaining ties are broken by fewer
#' false blocks, then fewer false approves, then a larger approve band. This
#' makes the result deterministic.
#'
#' ## Complexity
#'
#' With \eqn{n} units the selection runs in \eqn{O(n \log n)} time, dominated
#' by the sort, and \eqn{O(n)} memory. The batch is sorted by risk, equal risks
#' are collapsed into groups, and prefix counts of positive labels turn the
#' cost into two additive parts. A monotone sweep over the upper cut point
#' maintains the best lower cut point in a sliding window with a deque, so no
#' pair of cut points is ever enumerated. Computing the risk itself with PSIS
#' and [E_loo()] is separate and costs the usual PSIS-LOO amount.
#'
#' @export
#' @param x For the matrix method, a numeric matrix of posterior draws of the
#'   event probability, with one row per draw and one column per unit (the
#'   same layout as `x` in [loo_predictive_metric()]). For the numeric method,
#'   a vector of leave-one-out risk probabilities, one per unit, for example
#'   the `value` returned by `E_loo(..., type = "mean")`.
#' @param y A numeric or logical vector of observed binary outcomes, `1` for
#'   the adverse event and `0` otherwise, with one element per unit.
#' @param log_lik A matrix of pointwise log-likelihoods with the same
#'   dimensions as `x`.
#' @param ... Additional arguments passed to the numeric method (for the
#'   matrix method) or ignored.
#' @param cost_false_approve Non-negative finite cost of approving a unit with
#'   \eqn{y_i = 1}.
#' @param cost_false_block Non-negative finite cost of blocking a unit with
#'   \eqn{y_i = 0}.
#' @param cost_review Non-negative finite cost of reviewing one unit.
#' @param max_reviews Maximum number of units that may be sent to review. A
#'   non-negative whole number or `Inf` (the default).
#' @param max_false_block Maximum number of units with \eqn{y_i = 0} that may be
#'   blocked. A non-negative whole number or `Inf` (the default).
#' @param r_eff A vector of relative effective sample size estimates containing
#'   one element per unit. See [psis()] for details.
#' @param cores The number of cores to use for parallelization of [psis()].
#'   See [psis()] for details.
#' @param psis_object Optionally, a precomputed `"psis"` object for `log_lik`
#'   (for example from `loo(..., save_psis = TRUE)`). If `NULL` (the default)
#'   it is computed internally.
#' @param digits For the print method, the number of significant digits used
#'   when printing the thresholds and costs.
#'
#' @return An object of class `"loo_decision_thresholds"`, a named list with
#'   components:
#' \describe{
#'   \item{`thresholds`}{
#'     Named numeric vector `c(low, high)` for the decision rule above. `low`
#'     and `high` are the reusable cut points.
#'   }
#'   \item{`counts`}{
#'     Named integer vector with the number of units in each band
#'     (`approve`, `review`, `block`), the numbers of `false_approve` and
#'     `false_block` decisions, the numbers of reviewed benign and adverse
#'     units (`review_negative`, `review_positive`), and the correctly decided
#'     benign and adverse counts (`true_approve`, `true_block`).
#'   }
#'   \item{`cost`}{
#'     Named numeric vector with the cost split into `false_approve`,
#'     `false_block`, `review`, and `total`.
#'   }
#'   \item{`decisions`}{
#'     Character vector with the chosen band (`"approve"`, `"review"`, or
#'     `"block"`) for each unit, in the input order.
#'   }
#'   \item{`risk`}{
#'     Numeric vector of the risk probabilities used, in the input order.
#'   }
#'   \item{`n`}{The number of units.}
#'   \item{`pareto_k`}{
#'     Matrix method only: the function-specific Pareto \eqn{k} diagnostics
#'     returned by [E_loo()].
#'   }
#'   \item{`call`}{The matched call.}
#' }
#'
#' @seealso [loo_predictive_metric()] for the usual leave-one-out binary
#'   classification metrics, [E_loo()] for the weighted expectations used to
#'   compute the risk, and [psis()] for the underlying approximation.
#'
#' @examples
#' LL <- example_loglik_matrix()
#' set.seed(123)
#' # Posterior draws of the event probability and binary outcomes
#' prob_draws <- plogis(matrix(rnorm(length(LL)), nrow = nrow(LL)))
#' y <- rbinom(ncol(LL), size = 1, prob = 0.4)
#'
#' fit <- loo_decision_thresholds(
#'   prob_draws, y, LL,
#'   cost_false_approve = 10,
#'   cost_false_block = 2,
#'   cost_review = 0.5,
#'   max_reviews = 10,
#'   max_false_block = 2
#' )
#' fit
#' # The thresholds can be applied to new risk probabilities
#' new_risk <- runif(5)
#' ifelse(new_risk < fit$thresholds[["low"]], "approve",
#'        ifelse(new_risk < fit$thresholds[["high"]], "review", "block"))
loo_decision_thresholds <- function(x, ...) {
  UseMethod("loo_decision_thresholds")
}

#' @rdname loo_decision_thresholds
#' @export
loo_decision_thresholds.numeric <-
  function(x,
           y,
           ...,
           cost_false_approve,
           cost_false_block,
           cost_review,
           max_reviews = Inf,
           max_false_block = Inf) {
    risk <- x
    y <- .assert_binary_outcome(y, length(risk))
    .assert_decision_input(risk,
                           cost_false_approve,
                           cost_false_block,
                           cost_review,
                           max_reviews,
                           max_false_block)

    out <- .decision_thresholds(
      risk = risk,
      y = y,
      cost_false_approve = cost_false_approve,
      cost_false_block = cost_false_block,
      cost_review = cost_review,
      max_reviews = max_reviews,
      max_false_block = max_false_block
    )
    out$call <- match.call()
    class(out) <- "loo_decision_thresholds"
    out
  }

#' @rdname loo_decision_thresholds
#' @export
loo_decision_thresholds.matrix <-
  function(x,
           y,
           log_lik,
           ...,
           cost_false_approve,
           cost_false_block,
           cost_review,
           max_reviews = Inf,
           max_false_block = Inf,
           r_eff = 1,
           cores = getOption("mc.cores", 1),
           psis_object = NULL) {
    stopifnot(
      is.numeric(x),
      identical(ncol(x), length(y)),
      identical(dim(x), dim(log_lik)),
      is.null(psis_object) ||
        (is.psis(psis_object) && identical(dim(psis_object), dim(x)))
    )
    if (is.null(psis_object)) {
      psis_object <- psis(-log_lik, r_eff = r_eff, cores = cores)
    }
    e_loo <- E_loo(
      x,
      psis_object = psis_object,
      log_ratios = -log_lik,
      type = "mean"
    )
    risk <- as.vector(e_loo$value)
    out <- loo_decision_thresholds.numeric(
      x = risk,
      y = y,
      cost_false_approve = cost_false_approve,
      cost_false_block = cost_false_block,
      cost_review = cost_review,
      max_reviews = max_reviews,
      max_false_block = max_false_block
    )
    out$pareto_k <- e_loo$pareto_k
    out$call <- match.call()
    out
  }

#' @rdname loo_decision_thresholds
#' @export
print.loo_decision_thresholds <- function(x, digits = 3, ...) {
  cat("\nCost-optimal two-threshold decisions\n")
  cat("  approve if risk < ", .fr(x$thresholds[["low"]], digits), "\n", sep = "")
  cat("  review  if ",
      .fr(x$thresholds[["low"]], digits),
      " <= risk < ",
      .fr(x$thresholds[["high"]], digits),
      "\n",
      sep = "")
  cat("  block   if risk >= ",
      .fr(x$thresholds[["high"]], digits),
      "\n\n",
      sep = "")
  counts <- x$counts[c("approve", "review", "block")]
  print(data.frame(decision = names(counts), n = unname(counts)),
        row.names = FALSE)
  cat("\nCost:\n")
  cost <- x$cost
  print(data.frame(component = names(cost), cost = unname(cost)),
        row.names = FALSE, digits = digits)
  invisible(x)
}


# ----------------------------- Internals -----------------------------

#' Relative tolerance used for cost ties
#'
#' @noRd
#' @param a,b Numeric scalars.
#' @return `TRUE` if `a` and `b` are equal up to floating point error.
.cost_tied <- function(a, b) {
  abs(a - b) <= 1e-12 * max(1, abs(a), abs(b))
}

#' Compare two candidate decision rules
#'
#' @noRd
#' @param candidate,best Lists with components `cost`, `n_review`,
#'   `n_false_block`, `n_false_approve`, `i`, and `j`.
#' @return `TRUE` if `candidate` should replace `best`.
.rule_better <- function(candidate, best) {
  if (is.null(best)) {
    return(TRUE)
  }
  if (!.cost_tied(candidate$cost, best$cost)) {
    return(candidate$cost < best$cost)
  }
  if (candidate$n_review != best$n_review) {
    return(candidate$n_review < best$n_review)
  }
  if (candidate$n_false_block != best$n_false_block) {
    return(candidate$n_false_block < best$n_false_block)
  }
  if (candidate$n_false_approve != best$n_false_approve) {
    return(candidate$n_false_approve < best$n_false_approve)
  }
  if (candidate$i != best$i) {
    return(candidate$i > best$i)
  }
  candidate$j > best$j
}

#' Validate a binary outcome vector
#'
#' @noRd
#' @param y Numeric or logical vector.
#' @param n Expected length.
#' @return `y` as a numeric vector of zeros and ones.
.assert_binary_outcome <- function(y, n) {
  if (is.logical(y)) {
    y <- as.numeric(y)
  }
  checkmate::assert_numeric(y, any.missing = FALSE, finite = TRUE, len = n)
  if (!all(y %in% c(0, 1))) {
    stop("`y` must contain only 0 and 1 (1 is the adverse outcome).",
         call. = FALSE)
  }
  y
}

#' Validate decision inputs
#'
#' @noRd
#' @param risk,cost_false_approve,cost_false_block,cost_review,max_reviews,max_false_block
#'   See [loo_decision_thresholds()].
#' @return `NULL`, invisibly. Errors if an input is invalid.
.assert_decision_input <- function(risk,
                                   cost_false_approve,
                                   cost_false_block,
                                   cost_review,
                                   max_reviews,
                                   max_false_block) {
  checkmate::assert_numeric(risk,
                            any.missing = FALSE,
                            finite = TRUE,
                            lower = 0,
                            upper = 1)
  .assert_cost(cost_false_approve, "cost_false_approve")
  .assert_cost(cost_false_block, "cost_false_block")
  .assert_cost(cost_review, "cost_review")
  .assert_count_or_inf(max_reviews, "max_reviews")
  .assert_count_or_inf(max_false_block, "max_false_block")
  invisible(NULL)
}

#' Validate a single cost argument
#'
#' @noRd
#' @param x Value to check.
#' @param name Argument name used in the error message.
#' @return `NULL`, invisibly.
.assert_cost <- function(x, name) {
  if (!checkmate::test_number(x, lower = 0, finite = TRUE)) {
    stop("`", name, "` must be a single finite non-negative number.",
         call. = FALSE)
  }
  invisible(NULL)
}

#' Validate a non-negative whole number or `Inf`
#'
#' @noRd
#' @param x Value to check.
#' @param name Argument name used in the error message.
#' @return `NULL`, invisibly.
.assert_count_or_inf <- function(x, name) {
  if (!checkmate::test_number(x, lower = 0, finite = FALSE)) {
    stop("`", name, "` must be a single non-negative number or Inf.",
         call. = FALSE)
  }
  if (!is.infinite(x) && x != floor(x)) {
    stop("`", name, "` must be a whole number or Inf.", call. = FALSE)
  }
  invisible(NULL)
}

#' Select the cost-optimal pair of cut points
#'
#' Implements the O(n log n) selection described in
#' [loo_decision_thresholds()]. Equal risks are collapsed into groups so that
#' a cut point never splits tied units. For group cut points \eqn{i \le j} the
#' total cost is `A[i] + B[j]`, and the review bound makes the admissible
#' \eqn{i} a suffix of `0:j` whose left end only moves right as \eqn{j} grows.
#' A monotone deque then tracks the best \eqn{i} for every \eqn{j} without
#' enumerating cut point pairs.
#'
#' @noRd
#' @param risk Numeric vector of risk probabilities in `[0, 1]`.
#' @param y Numeric vector of zeros and ones of the same length.
#' @param cost_false_approve,cost_false_block,cost_review,max_reviews,max_false_block
#'   See [loo_decision_thresholds()].
#' @return An unclassed list with the components documented in
#'   [loo_decision_thresholds()] except `call` and `pareto_k`.
.decision_thresholds <- function(risk,
                                 y,
                                 cost_false_approve,
                                 cost_false_block,
                                 cost_review,
                                 max_reviews,
                                 max_false_block) {
  n <- length(risk)
  ord <- order(risk)
  risk_sorted <- risk[ord]
  y_sorted <- y[ord]

  # Collapse identical risks into groups. `rle()` on the sorted values gives
  # the distinct risk values and their sizes, so cut points can only fall
  # between groups and tied units always share a decision.
  groups <- rle(risk_sorted)
  sizes <- groups$lengths
  scores <- groups$values
  m <- length(sizes)
  group_id <- rep.int(seq_len(m), sizes)

  # Prefix counts over groups (index k corresponds to groups 1..k).
  ends <- cumsum(sizes)
  pos_ends <- cumsum(y_sorted)[ends]
  n_units <- c(0L, ends)
  n_pos <- c(0L, pos_ends)
  n_neg <- n_units - n_pos
  total_neg <- n_neg[m + 1L]

  # Cost(i, j) = A[i] + B[j], with groups 1..i approved, i+1..j reviewed,
  # and j+1..m blocked. Reviewed units cost only `cost_review` because a
  # review is assumed to be decided correctly.
  A <- cost_false_approve * n_pos - cost_review * n_units
  B <- cost_review * n_units +
    cost_false_block * (total_neg - n_neg)

  # Upper cut points that already violate the false-block bound can be
  # skipped. `n_neg` is non-decreasing in j, so `false_block` is non-increasing
  # and the feasible upper cut points form a suffix of 0:m. Blocking nothing
  # (j = m) is always feasible, so the suffix is never empty.
  false_block <- total_neg - n_neg
  first_feasible_j <- which(false_block <= max_false_block)[[1L]] - 1L

  # Monotone deque of candidate lower cut points. It keeps the admissible
  # `i` that minimizes A[i]; on A-ties the larger `i` (fewer reviews) is
  # preferred, hence the tolerance-aware pop from the back.
  deque <- integer(m + 1L)
  head <- 1L
  tail <- 0L
  i_min <- 0L
  best <- NULL

  for (j in 0:m) {
    a_j <- A[j + 1L]
    while (tail >= head) {
      a_back <- A[deque[tail] + 1L]
      if (a_back > a_j || .cost_tied(a_back, a_j)) {
        tail <- tail - 1L
      } else {
        break
      }
    }
    tail <- tail + 1L
    deque[tail] <- j

    # Keep i in the review window: n_units[j + 1] - n_units[i + 1] <= R.
    # The left end only moves right as j grows.
    need <- n_units[j + 1L] - max_reviews
    while (i_min < j && n_units[i_min + 1L] < need) {
      i_min <- i_min + 1L
    }
    while (head <= tail && deque[head] < i_min) {
      head <- head + 1L
    }

    if (j < first_feasible_j) {
      next
    }

    i <- deque[head]
    candidate <- list(
      i = i,
      j = j,
      cost = A[i + 1L] + B[j + 1L],
      n_review = n_units[j + 1L] - n_units[i + 1L],
      n_false_block = false_block[j + 1L],
      n_false_approve = n_pos[i + 1L]
    )
    if (.rule_better(candidate, best)) {
      best <- candidate
    }
  }

  i <- best$i
  j <- best$j

  # Half-open rule: approve below `low`, review `[low, high)`, block at or
  # above `high`. The cut points sit between distinct risk values.
  low <- if (i < m) scores[[i + 1L]] else Inf
  high <- if (j < m) scores[[j + 1L]] else Inf

  tier <- ifelse(group_id <= i, 1L, ifelse(group_id <= j, 2L, 3L))
  decisions_sorted <- c("approve", "review", "block")[tier]
  decisions <- character(n)
  decisions[ord] <- decisions_sorted

  n_approve <- n_units[i + 1L]
  n_review <- best$n_review
  n_block <- n - n_units[j + 1L]
  n_false_approve <- n_pos[i + 1L]
  n_false_block <- best$n_false_block
  n_review_pos <- n_pos[j + 1L] - n_pos[i + 1L]

  counts <- c(
    approve = n_approve,
    review = n_review,
    block = n_block,
    false_approve = n_false_approve,
    false_block = n_false_block,
    true_approve = n_approve - n_false_approve,
    true_block = n_block - n_false_block,
    review_negative = n_review - n_review_pos,
    review_positive = n_review_pos
  )
  counts <- setNames(as.integer(counts), names(counts))

  cost <- c(
    false_approve = cost_false_approve * n_false_approve,
    false_block = cost_false_block * n_false_block,
    review = cost_review * n_review
  )
  cost <- c(cost, total = sum(cost))

  list(
    thresholds = c(low = low, high = high),
    counts = counts,
    cost = cost,
    decisions = decisions,
    risk = risk,
    n = n
  )
}
