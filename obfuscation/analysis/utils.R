# Format ------------------------------------------------------------------
get_model_str <- function(model_name) {
  case_when(
    str_detect(model_name, "cascade") ~ "Cascade R-CNN",
    str_detect(model_name, "faster") ~ "Faster R-CNN",
    str_detect(model_name, "retina") ~ "RetinaNet",
    str_detect(model_name, "ssd") ~ "SSD",
    str_detect(model_name, "yolo") ~ "YOLOv3",
  )
}

get_target_str <- function(target_fun) {
  case_when(
    str_detect(target_fun, "mislabel") ~ "Mislabeling",
    str_detect(target_fun, "vanish") ~ "Vanishing",
    str_detect(target_fun, "untarget") ~ "Untargeted",
  )
}

wrangle_model_col <- function(model_col) {
  factor(get_model_str(model_col),
    levels = c("YOLOv3", "SSD", "RetinaNet", "Faster R-CNN", "Cascade R-CNN"),
    ordered = TRUE
  )
}

wrangle_target_col <- function(target_col) {
  factor(get_target_str(target_col),
    levels = c("Vanishing", "Mislabeling", "Untargeted"),
    ordered = TRUE
  )
}


wrangle_target_perturb_col <- function(target_perturb_col) {
  factor(recode(
    target_perturb_col,
    perturb = "Perturb", target = "Target"
  ), ordered = TRUE)
}

retain_upper <- function(text) {
  str_replace_all(text, regex(c(
    " IOU " = " IOU ",
    " COCO " = " COCO ",
    "Table " = "Table ",
    "Figure " = "Figure ",
    "Cascade R-CNN" = "Cascade R-CNN",
    "Faster R-CNN" = "Faster R-CNN",
    "RetinaNet" = "RetinaNet",
    "SSD" = "SSD",
    "YOLOv3" = "YOLOv3"
  ), ignore_case = TRUE))
}

round_table <- function(df, digits = 3) {
  df |> mutate(across(where(is.numeric), ~ round(., 3)))
}

# Compute -----------------------------------------------------------------
linear_space <- function(vec, n = 100) {
  seq(min(vec), max(vec), length.out = n)
}

include_target_perturb <- function(data) {
  "target_or_perturb" %in% colnames(data) &&
    length(unique(data$target_or_perturb)) == 2
}

# https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares
get_min_distance <-
  function(x1_1,
           y1_1,
           x2_1,
           y2_1,
           x1_2,
           y1_2,
           x2_2,
           y2_2) {
    # larger rectangle enveloping both
    x1 <- min(x1_1, x1_2)
    y1 <- min(y1_1, y1_2)
    x2 <- max(x2_1, x2_2)
    y2 <- max(y2_1, y2_2)

    inner_width <- max(0, (x2 - x1) - (x2_1 - x1_1) - (x2_2 - x1_2))
    inner_height <- max(0, (y2 - y1) - (y2_1 - y1_1) - (y2_2 - y1_2))
    min_distance <- sqrt(inner_width^2 + inner_height^2)
  }

# Wrangle -----------------------------------------------------------------
cols_equal <- function(data) {
  length(unique(as.list(data))) == 1
}

cols_start_equal <- function(data, cols_start) {
  d <- as_tibble(data)
  for (s in cols_start) {
    print(glue("Columns starting with `{s}` are equal: {d |>
      select(starts_with(s)) |>
      cols_equal()}"))
  }
}

get_data <- function(fnames, read_f, workers = availableCores() - 2, .id = "fname", show_col_types = FALSE, .progress = TRUE) {
  plan(multisession, workers = workers)

  future_map_dfr(fnames, read_f, .id = .id, show_col_types = show_col_types, .progress = .progress) |>
    mutate(
      model_name = wrangle_model_col(model_name),
      loss_target = wrangle_target_col(loss_target),
    )
}

# Gather target and perturb predicted bboxes
# Determine whether attack succeeded
# Restrict to largest attack iterations
wrangle_success <- function(data) {
  data |>
    # Column names are target and perturb tags distinguished by attack iterations
    # and values are whether the bbox is targeted or perturbed in the attack iteration.
    pivot_longer(
      cols = matches("^bbox_(target|perturb)_\\d+$"),
      names_to = "target_or_perturb_itr",
      values_to = "target_or_perturb_boolean"
    ) |>
    filter(target_or_perturb_boolean == TRUE) |>
    separate(target_or_perturb_itr,
      c(NA, "target_or_perturb", "attack_itr"),
      sep = "_", convert = TRUE
    ) |>
    # Restrict to target and perturb bboxes and
    # largest attack iterations to understand max impact trends
    filter(attack_itr == max(attack_itr)) |>
    mutate(target_or_perturb = wrangle_target_perturb_col(target_or_perturb)) |>
    # Column names are success tags distinguished by attack iterations
    # and values are whether the bbox belongs to a sample which succeeded in the attack iteration
    pivot_longer(
      cols = matches("^sample_success_\\d+$"),
      names_to = "success_itr",
      values_to = "success"
    ) |>
    # Success columns may not exist in simulations with zero success
    # and upon merging these columns will contain only NAs
    mutate(
      success_itr = as.numeric(str_extract(success_itr, "\\d+")),
      success = ifelse(replace_na(success, FALSE), 1, 0)
    ) |>
    # Determine whether target and perturb bboxes succeeded
    # by removing perturb or target iterations which don't match success iteration
    filter(attack_itr == success_itr) |>
    select(-success_itr)
}

# Graph ----------------------------------------------------------------
set_theme <- function(base = theme_minimal) {
  theme_set(base())
  theme_update(
    text = element_text(size = 10),
    legend.position = "top",
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.line.x = element_line(colour = "grey87"),
    axis.ticks.x = element_line(colour = "grey87"),
  )
}

# default is y ~ x
binomial_smooth <- function(...) {
  # Error is 95% CI by default
  # https://ggplot2.tidyverse.org/reference/geom_smooth.html
  geom_smooth(method = "glm", method.args = list(family = "binomial"), ...)
}

# selected columns should have no NAs
check_graph_data <- function(data, variable) {
  if (include_target_perturb(data)) {
    d <- data |>
      select(fname, model_name, loss_target, {{ variable }}, success, target_or_perturb, attack_itr)
  } else {
    d <- data |>
      select(fname, model_name, loss_target, {{ variable }}, success, attack_itr)
  }

  stopifnot(sum(is.na(d)) == 0)
}

# Graph success against attribute with model as columns and attack types as rows,
# potentially grouped by perturb and target bboxes
graph_attr <- function(data, variable, x_label, gg_options = NULL) {
  # variable is a symbol not a string
  # https://stackoverflow.com/a/48062317/19655086
  # https://rlang.r-lib.org/reference/topic-data-mask.html
  check_graph_data(data, {{ variable }})

  if (!include_target_perturb(data)) {
    g <- data |>
      ggplot(aes({{ variable }}, success))
  } else {
    g <- data |>
      ggplot(aes({{ variable }}, success, color = target_or_perturb)) +
      labs(color = "Object")
  }

  breaks <- data |>
    pull({{ variable }}) |>
    quantile()

  # g <- g + stat_summary_bin(fun.data = "mean_cl_boot", bins = 5) +
  g <- g + stat_summary_bin(fun.data = "mean_cl_boot", breaks = breaks) +
    binomial_smooth(formula = y ~ x) +
    facet_grid(rows = vars(loss_target), cols = vars(model_name)) +
    labs(x = retain_upper(str_to_title(x_label)), y = "p(Success)") +
    guides(x = guide_axis(angle = 90))

  if (is.null(gg_options)) {
    # https://stackoverflow.com/a/27201707/19655086
    # https://ggplot2.tidyverse.org/reference/print.ggplot.html
    print(g)
  } else {
    print(g + gg_options)
  }
}

bold_tex <- function(text) {
  glue("\\textbf{{{text}:}} ")
}

err_cap <- "Errors are 95\\% confidence intervals. "

bin_sum <- "Bins are split into quantiles. "

graph_caption <- function(pred_name, main_pt, exp_name = "randomized") {
  retain_upper(glue(
    "{bold_tex(str_to_sentence(main_pt))} The binned summaries and regression trendlines",
    " graph success proportion against {str_to_lower(pred_name)} in the",
    " {exp_name} attack experiment. ", bin_sum, err_cap
  ))
}

# Table -----------------------------------------------------------------
table_caption <- function(pred_name, main_pt, exp_name = "randomized") {
  retain_upper(glue(
    "We run a logistic model regressing success against {str_to_lower(pred_name)}",
    " in the {exp_name} attack experiment. {str_to_sentence(main_pt)}.",
    " Table headers are explained in Appendix \\ref{{app:tab_hdr}}."
  ))
}

rename_col <- function(col_name) {
  case_when(
    col_name == "model_name" ~ "Model",
    col_name == "loss_target" ~ "Attack",
    col_name == "label" ~ "term",
    TRUE ~ col_name
  )
}

sub_col <- function(col) {
  # discrete => plural
  str_replace_all(col, coll(c(
    "bbox_dist" = "distance",
    "bbox_len" = "length",
    "num_iteration" = "iterations",
    "bbox_conf" = "confidence",
    "bbox_size_perturb" = "size",
    "gt_p_success" = "accuracy",
    "sample_mislabel_proba" = "probability",
    "bbox_iou_predictions_eval" = "iou"
  )))
}

print_statistics <- function(data, caption, alpha = 0.05) {
  opts <- options(knitr.kable.NA = "")

  d <- data |>
    ungroup() |>
    select(any_of(c("model_name", "loss_target")) | c(label, estimate:conf.high)) |>
    mutate(
      label = sub_col(label),
      sig = if_else(p.value < alpha, "*", "")
    ) |>
    relocate(sig, .after = label)

  n_grp_cols <- ncol(select(d, any_of(c("model_name", "loss_target"))))
  n_reg_cols <- ncol(d) - n_grp_cols

  if (n_grp_cols == 2 && length(unique(d$loss_target)) == 1) {
    d <- d |> relocate(loss_target, .before = model_name)
  }

  t <- d |>
    rename_with(rename_col) |>
    kbl(
      booktabs = TRUE,
      longtable = TRUE,
      caption = caption,
      position = "tb",
      digits = 3
    ) |>
    kable_styling(
      position = "center",
      font_size = 9
    ) |>
    add_header_above(c("Group" = n_grp_cols, "Regression" = n_reg_cols))

  if (n_grp_cols == 2) {
    t |> collapse_rows(columns = 1:2, row_group_label_position = "stack")
  } else {
    t |> collapse_rows(columns = 1)
  }
}

# Regression --------------------------------------------------------------
glm_model <- function(data, predictor) {
  glm(as.formula(glue("success ~ {predictor}")), family = "binomial", data = data)
}

get_tidied_reg <- function(model, data, grp_vars = c(model_name, loss_target), return_mod = FALSE) {
  stopifnot(!include_target_perturb(data))

  # https://stackoverflow.com/questions/57704792/is-it-possible-to-pass-multible-variables-to-the-same-curly-curly
  mod_data <- data |>
    ungroup() |>
    nest_by(across({{ grp_vars }})) |>
    mutate(mod = list(model(data)))

  tidied_data <- mod_data |>
    summarize(tidy_plus_plus(mod, conf.int = TRUE))

  if (return_mod) {
    list(mod = mod_data, tidied = tidied_data)
  } else {
    tidied_data
  }
}

ext_sig <- function(reg_est, direction = "both", var_name = NULL, alpha = 0.05) {
  stopifnot(direction %in% c("pos", "neg", "both"))

  reg_est <- reg_est |> select(any_of(c("model_name", "loss_target")) | c(term, estimate:conf.high))
  if (is.null(var_name)) {
    reg_vars <- reg_est
  } else {
    print(glue("----------{var_name}----------"))
    reg_vars <- reg_est |> filter(term == var_name)
  }

  reg_sig <- reg_vars |> filter(p.value < alpha)
  p_sig <- nrow(reg_sig) / nrow(reg_vars)

  if (direction == "pos") {
    reg_dir <- reg_sig |> filter(estimate > 0)
  } else if (direction == "neg") {
    reg_dir <- reg_sig |> filter(estimate < 0)
  } else {
    reg_dir <- reg_sig
  }

  p_dir <- nrow(reg_dir) / nrow(reg_vars)
  print(glue("Total {nrow(reg_vars)} predictors:
             {nrow(reg_sig)} ({round(p_sig * 100)}%) significant;
             {nrow(reg_dir)} ({round(p_dir * 100)}%) {direction}"))

  round_table(reg_dir)
}
