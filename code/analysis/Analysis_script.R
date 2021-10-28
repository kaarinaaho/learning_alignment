library(tidyr)
library(dplyr)
library(tibble)
library(ez)
library(apaTables)

# Import filtered data: ----------------------------------------------------------#

setwd('..')
setwd('..')
wd <- getwd()

data <- as.data.frame(read.csv("./data/experiment/processed/Processed_data_filtered.csv",
                               header=TRUE, row.names=1,
                               stringsAsFactors = TRUE)
                      )

# Run main analysis: -------------------------------------------------------------#

# Block means: factorial ANOVA of all --------------------------------------------#

block_groups <- data %>%
  filter(block < 5) %>%  # Filter out generalisation trial; blocks are zero-indexed
  group_by(align_condition, rotate_condition, block, pid) %>%
  summarise(correct = mean(correct), dist_from_correct = mean(dist_from_correct))

block_groups$block <- as.factor(block_groups$block)

# Repeated measures ANOVA - accuracy
rm.anova.acc.ez <- ezANOVA(
  data=block_groups,
  dv=.(correct),
  wid=.(pid),
  within=.(block),
  between=.(align_condition, rotate_condition),
  type=2
  )
print(rm.anova.acc.ez)
apa.ezANOVA.table(rm.anova.acc.ez, correction="HF") # Results table - accuracy

# Repeated measures ANOVA - dist from correct
rm.anova.dist.ez <- ezANOVA(
  data=block_groups,
  dv=.(dist_from_correct), 
  wid=.(pid), 
  within=.(block), 
  between=.(align_condition, rotate_condition),
  type=2
  )
print(rm.anova.dist.ez)
apa.ezANOVA.table(rm.anova.dist.ez, correction="HF") # Results table - dist error


# Generalisation performance ----------------------------------------------------#

# Chi-squared for generalisation performance

gen <- data %>%
       filter(block == 5) %>%
       filter(align_condition=="aligned")%>%
       rename(gen_acc = correct) %>%
       group_by(gen_acc) %>%
       summarise(count_pids = n_distinct(pid))

gen_chisq <- chisq.test(gen$count_pids)

# Chi-squared including rotation condition

gen_rot <- data %>%
           filter(block == 5) %>%
           filter(align_condition=="aligned")%>%
           rename(gen_acc = correct) %>%
           group_by(gen_acc, rotate_condition) %>%
           summarise(count_pids = n_distinct(pid))

gen_rot <- gen_rot %>%
           spread(key=rotate_condition, value=count_pids)

gen_rot.rownames <- gen_rot$gen_acc
gen_rot$gen_acc <- NULL

gen_rot_chisq <-chisq.test(gen_rot)


# Compare best model counts ----------------------------------------------------#

models <- as.data.frame(read.csv("./data/models/paper_results/fitted_model_hyperparams.csv",
                               header=TRUE, row.names=1,
                               stringsAsFactors = TRUE)
)

# Add random model
random <- distinct(models[c("pid", "align_condition")])
random$model <- "random"
random[c("lr_sup", "lr_unsup", "lr", "lam_a_cyc", "lam_dist", "hidden_size", "s", "alpha", "temp")] <- 0
random$loss <- -30 * log(1/6)

models <- rbind(models, random)


models$params <- 3
models$params[models$model == "cycle_and_distribution"] <- 5
models$params[models$model == "random"] <- 0
models$AIC <- 2*models["params"] + 2 * models["loss"]
models <- models %>%
          group_by(pid) %>%
          mutate(ranks = rank(AIC)) %>%
          ungroup()

# Select best fitting
models <- models %>%
          filter(ranks == 1)

# Count best fits by model type
best_models <- models %>%
               group_by(align_condition, model) %>%
               summarise(count_pids = n_distinct(pid))

best_models_aligned <- best_models %>%
                       filter(align_condition == "aligned")
best_models_aligned$pct <- best_models_aligned$count_pids/sum(best_models_aligned$count_pids)
chisq_bestmod_aligned <- chisq.test(best_models_aligned$count_pids)

best_models_misaligned <- best_models %>%
                          filter(align_condition == "misaligned")
best_models_misaligned$pct <- best_models_misaligned$count_pids/sum(best_models_misaligned$count_pids)
chisq_bestmod_misaligned <- chisq.test(best_models_misaligned$count_pids)

# ---------------------------------------------------------------------------------------------------------------------#


# Supplementary analyses: -------------------------------------------------------------#


# Entropy correlation and threshold
unf <- as.data.frame(read.csv("./data/experiment/processed/Processed_data.csv",
                               header=TRUE, row.names=1,
                               stringsAsFactors = TRUE)
)

# Correlation of entropy and final block accuracy
for_corr <- unf %>%
            filter(block == 4) %>% # Blocks zero-indexed
            group_by(pid) %>%
            select(correct, entropy)%>%
            summarise(avg_acc = mean(correct), avg_ent = mean(entropy))

corr <- cor.test(for_corr$avg_acc, for_corr$avg_ent, method="pearson")
q <- quantile(for_corr$avg_ent, probs=0.1)

# Filtered
filt <- unf %>%
        filter(entropy >= q) %>%
        filter(age < 90)
  
# Chi-square for pre- and post- exclusion numbers
unf_cond <- unf %>%
            group_by(align_condition, rotate_condition) %>%
            summarise(count_pids = n_distinct(pid))
unf_cond$pct <- unf_cond$count_pids / sum(unf_cond$count_pids)

filt_cond <- filt %>%
             group_by(align_condition, rotate_condition) %>%
             summarise(count_pids = n_distinct(pid))

chisq_unf <- chisq.test(unf_cond$count_pids)
chisq_filt <- chisq.test(filt_cond$count_pids)
tbl <- cbind(filt_cond$count_pids, unf_cond$count_pids)
chisq_change <- chisq.test(tbl)

# Demographics
age_sum <- data %>%
  summarise(
    min_age = min(age),
    max_age = max(age),
    mean_age = mean(age),
    sd_age = sd(age)
  )

gender <- data %>%
  group_by(gender) %>%
  summarise(
    n_distinct(pid)
  )


# Participant distributions before/after exclusions
filt$condition <- paste(filt$align_condition, filt$rotate_condition)
filt_age <- filt %>%
            select(age, condition, pid)
filt_age <- distinct(filt_age)
age.aov <- aov(age ~ condition, data = filt_age)
summary(age.aov)

# Chisq for female proportions between exclusion states
filt_gen <- filt %>%
            group_by(gender) %>%
            summarise(count_pids = n_distinct(pid))

unf_gen <- unf %>%
            group_by(gender) %>%
            summarise(count_pids = n_distinct(pid))

tbl <- cbind(filt_gen$count_pids, unf_gen$count_pids)
chisq_gend_change <- chisq.test(tbl)


# Generalisation in misaligned condition
gen_mis <- data %>%
  filter(block == 5) %>%
  filter(align_condition=="misaligned")%>%
  rename(gen_acc = correct) %>%
  group_by(gen_acc) %>%
  summarise(count_pids = n_distinct(pid))

gen_mis_chisq <- chisq.test(gen_mis$count_pids)










