rm(list = ls())

setwd("~/School/TU/Graduate/17th Grade/Spring/Evolutionary Computation/Project")

library(dplyr)
library(tidyverse)
library(readxl)
library(caret)
library(ggforce)

# https://www.geeksforgeeks.org/how-to-read-a-xlsx-file-with-multiple-sheets-in-r/
multiplesheets <- function(fname) {
  sheets <- readxl::excel_sheets(fname)
  tib <- lapply(sheets, function(x) readxl::read_excel(fname, sheet = x))
  df <- lapply(tib, as.data.frame)
  names(df) <- sheets
  df
}

df <- multiplesheets("results.xlsx")

hc <- df$HC
sa <- df$SA
ga <- df$GA

hc <- hc %>% arrange(`# of Genes`, Lopsidedness, Perturbation)
sa <- sa %>% arrange(`# of Genes`, Lopsidedness, Perturbation)
ga <- ga %>% arrange(`# of Genes`, Lopsidedness, Selection, Crossover, Mutation)

hc <- hc %>% select(-c(`# of Trials`, i0, beta))
cols_to_factor <- c("Perturbation")
hc[cols_to_factor] <- lapply(hc[cols_to_factor], factor)
cols_to_percent <- c("Avg % Difference", "Optimal Found %")
hc[cols_to_percent] <- hc[cols_to_percent] * 100
hc %>% glimpse()

sa <- sa %>% select(-c(`# of Trials`, t0, i0, alpha, beta))
cols_to_factor <- c("Perturbation")
sa[cols_to_factor] <- lapply(sa[cols_to_factor], factor)
cols_to_percent <- c("Avg % Difference", "Optimal Found %")
sa[cols_to_percent] <- sa[cols_to_percent] * 100
sa %>% glimpse()

ga <- ga %>% select(-c(`# of Trials`, `Max # of Gens`, `Pop Size`))
cols_to_factor <- c("Selection", "Crossover", "Mutation")
ga[cols_to_factor] <- lapply(ga[cols_to_factor], factor)
cols_to_percent <- c("Avg % Difference", "Optimal Found %")
ga[cols_to_percent] <- ga[cols_to_percent] * 100
ga %>% glimpse()

hc_perturbation <- hc %>% 
  group_by(Perturbation) %>% 
  summarize(Diff = mean(`Avg % Difference`),
            Optm = mean(`Optimal Found %`),
            Loops = mean(`Avg # of Loops`),
            Time = mean(`Avg Time (s)`))

sa_perturbation <- sa %>% 
  group_by(Perturbation) %>% 
  summarize(Diff = mean(`Avg % Difference`),
            Optm = mean(`Optimal Found %`),
            Loops = mean(`Avg # of Loops`),
            Time = mean(`Avg Time (s)`))

ga_all <- ga %>% 
  group_by(Selection, Crossover, Mutation) %>% 
  summarize(Diff = mean(`Avg % Difference`),
            Optm = mean(`Optimal Found %`),
            Gens = mean(`Avg # of Gens`),
            Time = mean(`Avg Time (s)`))

ga_lopsided <- ga %>% 
  filter(Lopsidedness >= 0) %>% 
  group_by(Selection, Crossover, Mutation) %>% 
  summarize(Diff = mean(`Avg % Difference`),
            Optm = mean(`Optimal Found %`),
            Gens = mean(`Avg # of Gens`),
            Time = mean(`Avg Time (s)`))

#### Foolish Hill Climbing ####

hc %>% ggplot(aes(x = Lopsidedness, y = `Avg % Difference`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg % Difference vs. Lopsidedness (Foolish Hill Climbing)") +
  theme(legend.position = c(0.9, 0.83))
hc %>% ggplot(aes(x = Lopsidedness, y = `Avg # of Loops`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg # of Loops vs. Lopsidedness (Foolish Hill Climbing)") +
  theme(legend.position = c(0.9, 0.83))

hc %>% ggplot(aes(x = `# of Genes`, y = `Avg % Difference`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg % Difference vs. # of Genes (Foolish Hill Climbing)") +
  theme(legend.position = c(0.9, 0.83))
hc %>% ggplot(aes(x = `# of Genes`, y = `Avg # of Loops`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg # of Loops vs. # of Genes (Foolish Hill Climbing)") +
  theme(legend.position = c(0.1, 0.83))

hc %>% ggplot(aes(x = `Avg # of Loops`, y = `Avg % Difference`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg % Difference vs. Avg # of Loops (Foolish Hill Climbing)") +
  theme(legend.position = c(0.9, 0.83))

hc_perturbation

#### Simulated Annealing ####

sa %>% ggplot(aes(x = Lopsidedness, y = `Avg % Difference`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg % Difference vs. Lopsidedness (Simulated Annealing)") +
  theme(legend.position = c(0.9, 0.83))
sa %>% ggplot(aes(x = Lopsidedness, y = `Avg # of Loops`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg # of Loops vs. Lopsidedness (Simulated Annealing)") +
  theme(legend.position = c(0.9, 0.83))

sa %>% ggplot(aes(x = `# of Genes`, y = `Avg % Difference`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg % Difference vs. # of Genes (Simulated Annealing)") +
  theme(legend.position = c(0.9, 0.83))
sa %>% ggplot(aes(x = `# of Genes`, y = `Avg # of Loops`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg # of Loops vs. # of Genes (Simulated Annealing)") +
  theme(legend.position = c(0.1, 0.83))

sa %>% ggplot(aes(x = `Avg # of Loops`, y = `Avg % Difference`, col = Perturbation)) +
  geom_point() +
  ggtitle("Avg % Difference vs. Avg # of Loops (Simulated Annealing)") +
  theme(legend.position = c(0.9, 0.83))

sa_perturbation

#### Genetic Algorithm ####

ga %>% ggplot(aes(x = Lopsidedness, y = `Avg % Difference`, col = Selection, shape = Crossover)) +
  geom_point() +
  ggtitle("Avg % Difference vs. Lopsidedness (Genetic Algorithm)") +
  theme(legend.position = c(0.16, 0.64))
ga %>% ggplot(aes(x = Lopsidedness, y = `Avg # of Gens`, col = Selection, shape = Crossover)) +
  geom_point() +
  ggtitle("Avg # of Gens vs. Lopsidedness (Genetic Algorithm)") +
  theme(legend.position = c(0.16, 0.64))

ga %>% ggplot(aes(x = `# of Genes`, y = `Avg % Difference`, col = Selection, shape = Crossover)) +
  geom_point() +
  ggtitle("Avg % Difference vs. # of Genes (Genetic Algorithm)") +
  theme(legend.position = c(0.18, 0.82), legend.box = "horizontal")
ga %>% ggplot(aes(x = `# of Genes`, y = `Avg # of Gens`, col = Selection, shape = Crossover)) +
  geom_point() +
  ggtitle("Avg # of Gens vs. # of Genes (Genetic Algorithm)") +
  theme(legend.position = c(0.84, 0.36))

ga %>% ggplot(aes(x = `Avg Time (s)`, y = `Avg % Difference`, col = Selection, shape = Crossover)) +
  geom_point() +
  ggtitle("Avg % Difference vs. Avg Time (Genetic Algorithm)") +
  theme(legend.position = c(0.09, 0.69))

ga_all
ga_lopsided
