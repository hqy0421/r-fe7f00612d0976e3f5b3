LL <- example_loglik_matrix()
chain_id <- rep(1:2, each = dim(LL)[1] / 2)
r_eff <- relative_eff(exp(LL), chain_id)

set.seed(424)
prob_draws <- plogis(matrix(rnorm(length(LL)), nrow = nrow(LL)))
y_binary <- rbinom(ncol(LL), 1, 0.4)
risk <- colMeans(prob_draws)

# Brute-force reference over all contiguous (threshold-expressible) rules.
brute_thresholds <- function(risk, y, cfa, cfb, crev, R = Inf, F = Inf) {
  ord <- order(risk)
  s <- risk[ord]
  ys <- y[ord]
  groups <- rle(s)
  m <- length(groups$lengths)
  ends <- cumsum(groups$lengths)
  n_pos <- c(0, cumsum(ys)[ends])
  n_units <- c(0, ends)
  n_neg <- n_units - n_pos
  total_neg <- n_neg[m + 1]
  best <- NULL
  for (i in 0:m) {
    for (j in i:m) {
      n_review <- n_units[j + 1] - n_units[i + 1]
      n_fb <- total_neg - n_neg[j + 1]
      n_fa <- n_pos[i + 1]
      if (n_review > R || n_fb > F) next
      candidate <- list(
        i = i, j = j,
        cost = cfa * n_fa + crev * n_review + cfb * n_fb,
        n_review = n_review, n_false_block = n_fb, n_false_approve = n_fa
      )
      if (.rule_better(candidate, best)) best <- candidate
    }
  }
  best
}

test_that('loo_decision_thresholds stops with incorrect inputs', {
  expect_error(
    loo_decision_thresholds(c(-0.1, 0.5), c(0, 1), cost_false_approve = 1,
                            cost_false_block = 1, cost_review = 1),
    'risk'
  )
  expect_error(
    loo_decision_thresholds(c(0.2, 0.5), c(0, 2), cost_false_approve = 1,
                            cost_false_block = 1, cost_review = 1),
    'only 0 and 1'
  )
  expect_error(
    loo_decision_thresholds(c(0.2, 0.5), c(0, NA), cost_false_approve = 1,
                            cost_false_block = 1, cost_review = 1)
  )
  expect_error(
    loo_decision_thresholds(c(0.2, 0.5), c(0, 1), cost_false_approve = -1,
                            cost_false_block = 1, cost_review = 1),
    'cost_false_approve'
  )
  expect_error(
    loo_decision_thresholds(c(0.2, 0.5), c(0, 1), cost_false_approve = 1,
                            cost_false_block = Inf, cost_review = 1),
    'cost_false_block'
  )
  expect_error(
    loo_decision_thresholds(c(0.2, 0.5), c(0, 1), cost_false_approve = 1,
                            cost_false_block = 1, cost_review = 1,
                            max_reviews = 1.5),
    'whole number'
  )
  expect_error(
    loo_decision_thresholds(c(0.2, 0.5), c(0, 1), cost_false_approve = 1,
                            cost_false_block = 1, cost_review = 1,
                            max_false_block = -1),
    'max_false_block'
  )
})

test_that('loo_decision_thresholds returns the documented structure', {
  out <- loo_decision_thresholds(
    risk, y_binary,
    cost_false_approve = 10, cost_false_block = 2, cost_review = 0.5,
    max_reviews = 10, max_false_block = 2
  )
  expect_s3_class(out, 'loo_decision_thresholds')
  expect_named(out, c('thresholds', 'counts', 'cost', 'decisions', 'risk',
                      'n', 'call'))
  expect_named(out$thresholds, c('low', 'high'))
  expect_true(all(c('approve', 'review', 'block') %in% names(out$counts)))
  expect_named(out$cost,
               c('false_approve', 'false_block', 'review', 'total'))
  expect_length(out$decisions, length(risk))
  expect_true(all(out$decisions %in% c('approve', 'review', 'block')))
  expect_identical(out$n, length(risk))
  expect_true(all(out$counts[c('approve', 'review', 'block')] >= 0))
  expect_equal(sum(out$counts[c('approve', 'review', 'block')]), length(risk))
  expect_true(out$counts[['review']] <= 10)
  expect_true(out$counts[['false_block']] <= 2)
})

test_that('matrix method computes the same rule as the numeric method', {
  psis_obj <- psis(-LL, r_eff = r_eff, cores = 1)
  loo_risk <- as.vector(E_loo(prob_draws, psis_obj, log_ratios = -LL,
                              type = 'mean')$value)
  via_matrix <- loo_decision_thresholds(
    prob_draws, y_binary, LL,
    cost_false_approve = 5, cost_false_block = 1, cost_review = 0.25,
    max_reviews = 12, max_false_block = 3,
    r_eff = r_eff, cores = 1
  )
  via_numeric <- loo_decision_thresholds(
    loo_risk, y_binary,
    cost_false_approve = 5, cost_false_block = 1, cost_review = 0.25,
    max_reviews = 12, max_false_block = 3
  )
  expect_equal(via_matrix$thresholds, via_numeric$thresholds)
  expect_equal(via_matrix$counts, via_numeric$counts)
  expect_equal(via_matrix$cost, via_numeric$cost)
  expect_equal(via_matrix$decisions, via_numeric$decisions)
  expect_equal(via_matrix$risk, loo_risk)
  expect_length(via_matrix$pareto_k, ncol(prob_draws))

  via_psis <- loo_decision_thresholds(
    prob_draws, y_binary, LL,
    cost_false_approve = 5, cost_false_block = 1, cost_review = 0.25,
    max_reviews = 12, max_false_block = 3,
    psis_object = psis_obj
  )
  expect_equal(via_psis$counts, via_matrix$counts)
  expect_equal(via_psis$cost, via_matrix$cost)
})

test_that('costs and counts are internally consistent', {
  out <- loo_decision_thresholds(
    risk, y_binary,
    cost_false_approve = 7, cost_false_block = 3, cost_review = 1,
    max_reviews = Inf, max_false_block = Inf
  )
  dec <- out$decisions
  expect_equal(out$counts[['false_approve']], sum(dec == 'approve' & y_binary == 1))
  expect_equal(out$counts[['false_block']], sum(dec == 'block' & y_binary == 0))
  expect_equal(out$counts[['review']], sum(dec == 'review'))
  expect_equal(out$cost[['false_approve']], 7 * out$counts[['false_approve']])
  expect_equal(out$cost[['false_block']], 3 * out$counts[['false_block']])
  expect_equal(out$cost[['review']], 1 * out$counts[['review']])
  expect_equal(out$cost[['total']], sum(out$cost[1:3]))
})

test_that('thresholds reproduce the decisions and never split ties', {
  risk_ties <- sample(c(0.1, 0.1, 0.5, 0.5, 0.5, 0.9), 30, replace = TRUE)
  y_ties <- rbinom(30, 1, 0.5)
  out <- loo_decision_thresholds(
    risk_ties, y_ties,
    cost_false_approve = 4, cost_false_block = 2, cost_review = 1
  )
  applied <- ifelse(risk_ties < out$thresholds[['low']], 'approve',
                    ifelse(risk_ties < out$thresholds[['high']], 'review',
                           'block'))
  expect_identical(applied, unname(out$decisions))
  # Every distinct risk value maps to exactly one band.
  bands <- tapply(out$decisions, risk_ties, function(z) length(unique(z)))
  expect_true(all(bands == 1))
})

test_that('thresholds equal a brute-force search on random small problems', {
  set.seed(9)
  for (rep in seq_len(300)) {
    n <- sample(1:12, 1)
    risk_rep <- sample(c(0, 0.2, 0.5, 0.5, 0.8, 1), n, replace = TRUE)
    y_rep <- rbinom(n, 1, 0.5)
    cfa <- sample(c(0, 0.5, 1, 4, 10), 1)
    cfb <- sample(c(0, 0.5, 1, 4, 10), 1)
    crev <- sample(c(0, 0.25, 1, 3), 1)
    R <- sample(c(Inf, 0, 1, 2, 3, n), 1)
    F <- sample(c(Inf, 0, 1, 2, 3), 1)
    out <- loo_decision_thresholds(
      risk_rep, y_rep,
      cost_false_approve = cfa, cost_false_block = cfb, cost_review = crev,
      max_reviews = R, max_false_block = F
    )
    ref <- brute_thresholds(risk_rep, y_rep, cfa, cfb, crev, R, F)
    expect_equal(out$cost[['total']], ref$cost, tolerance = 1e-10)
    expect_equal(out$counts[['review']], ref$n_review)
    expect_equal(out$counts[['false_block']], ref$n_false_block)
    expect_equal(out$counts[['false_approve']], ref$n_false_approve)
  }
})

test_that('each band may be empty', {
  # All benign and cheap to approve: approve everything.
  out_approve <- loo_decision_thresholds(
    c(0.01, 0.02, 0.03), c(0, 0, 0),
    cost_false_approve = 100, cost_false_block = 5, cost_review = 1
  )
  expect_equal(out_approve$thresholds, c(low = Inf, high = Inf))
  expect_identical(unname(out_approve$counts[['review']]), 0L)
  expect_identical(unname(out_approve$counts[['block']]), 0L)

  # All adverse and cheap to block: block everything.
  out_block <- loo_decision_thresholds(
    c(0.4, 0.6, 0.9), c(1, 1, 1),
    cost_false_approve = 100, cost_false_block = 1, cost_review = 5
  )
  expect_identical(unname(out_block$counts[['review']]), 0L)
  expect_identical(unname(out_block$counts[['approve']]), 0L)
  expect_equal(out_block$thresholds[['low']], out_block$thresholds[['high']])

  # Expensive review forces an empty review band.
  out_no_review <- loo_decision_thresholds(
    c(0.05, 0.1, 0.9, 0.95), c(0, 0, 1, 1),
    cost_false_approve = 10, cost_false_block = 10, cost_review = 1000
  )
  expect_identical(unname(out_no_review$counts[['review']]), 0L)
  expect_equal(out_no_review$thresholds[['low']],
               out_no_review$thresholds[['high']])
})

test_that('constraints are respected and reported', {
  out <- loo_decision_thresholds(
    risk, y_binary,
    cost_false_approve = 100, cost_false_block = 1, cost_review = 0.01,
    max_reviews = 3
  )
  expect_true(out$counts[['review']] <= 3)

  out_fb <- loo_decision_thresholds(
    risk, y_binary,
    cost_false_approve = 100, cost_false_block = 0.1, cost_review = 0.01,
    max_false_block = 1
  )
  expect_true(out_fb$counts[['false_block']] <= 1)

  # Both constraints zero and no cost on approves: no review, no false block.
  out_zero <- loo_decision_thresholds(
    c(0.1, 0.2, 0.8, 0.9), c(0, 0, 1, 1),
    cost_false_approve = 10, cost_false_block = 10, cost_review = 1,
    max_reviews = 0, max_false_block = 0
  )
  expect_identical(unname(out_zero$counts[['review']]), 0L)
  expect_identical(unname(out_zero$counts[['false_block']]), 0L)
})

test_that('cost ties prefer fewer reviews', {
  # Two identical-risk units, one adverse: reviewing either gives the same
  # cost. The rule must not invent a review when approving/blocking is equal.
  out <- loo_decision_thresholds(
    c(0.5, 0.5), c(0, 1),
    cost_false_approve = 1, cost_false_block = 1, cost_review = 1
  )
  expect_identical(unname(out$counts[['review']]), 0L)
  # Splitting the tie is impossible, so both units share a band.
  expect_identical(unname(out$decisions[[1]]), unname(out$decisions[[2]]))
})

test_that('zero costs and Inf bounds are handled', {
  out <- loo_decision_thresholds(
    c(0.1, 0.4, 0.6, 0.9), c(0, 0, 1, 1),
    cost_false_approve = 0, cost_false_block = 0, cost_review = 0
  )
  expect_equal(out$cost[['total']], 0)
  expect_equal(sum(out$counts[c('approve', 'review', 'block')]), 4)
})

test_that('print method works', {
  out <- loo_decision_thresholds(
    risk, y_binary,
    cost_false_approve = 10, cost_false_block = 2, cost_review = 0.5
  )
  expect_output(print(out), 'Cost-optimal two-threshold decisions')
  expect_output(print(out), 'approve if risk <')
  expect_output(print(out), 'block')
  expect_invisible(print(out))
})

test_that('binary classification metrics are unchanged', {
  acc <- loo_predictive_metric(
    prob_draws, y_binary, LL, metric = 'acc', r_eff = r_eff, cores = 1
  )
  bacc <- loo_predictive_metric(
    prob_draws, y_binary, LL, metric = 'balanced_acc', r_eff = r_eff, cores = 1
  )
  expect_named(acc, c('estimate', 'se'))
  expect_named(bacc, c('estimate', 'se'))
})
