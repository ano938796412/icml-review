library(testthat)

source("./analysis/utils.R")

test_get_min_distance <- function() {
  bbox_1 <- c(1, 2, 2, 4)
  bbox_2s <- list(
    # bottom right
    list(1, 3, 2, 5),
    list(1, 4, 2, 6),
    list(1, 5, 2, 7),
    list(2, 4, 4, 5),
    list(2, 5, 4, 6),
    list(3, 4, 5, 5),
    list(3, 5, 5, 6),
    # top left
    list(2, 0, 4, 2),
    list(2, 1, 4, 3),
    list(2, -1, 4, 1),
    list(3, 0, 5, 2),
    list(3, -1, 5, 1)
  )
  exp_ans <- c( # bottom right
    0, 0, 1, 0, 1, 1, sqrt(2),
    # top left
    0, 0, 1, 1, sqrt(2)
  )

  for (i in seq_len(length(bbox_2s))) {
    expect_equal(do.call(get_min_distance, c(bbox_1, bbox_2s[[i]])), exp_ans[i])
    expect_equal(do.call(get_min_distance, c(bbox_2s[[i]], bbox_1)), exp_ans[i])
  }
}

test_get_min_distance()
