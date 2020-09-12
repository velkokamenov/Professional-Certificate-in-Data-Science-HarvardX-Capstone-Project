### The aim of this script is to create the model and calculate RMSE for MovieLens dataset ### 

# Author: Velko Kamenov

# The script takes around 8 minutes to be run

# This option disables scientifict notation for numbers visualization
options(scipen = 999)

# Install all needed for the project libraries if they are not found on the computer
if(!require(plyr)) install.packages("plyr")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(tinytex)) install.packages("tinytex")
if(!require(funModeling)) install.packages("funModeling")
if(!require(stringr)) install.packages("stringr")
if(!require(lubridate)) install.packages("lubridate")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(ggthemes)) install.packages("ggthemes")
if(!require(mltools)) install.packages("mltools")
if(!require(Hmisc)) install.packages("Hmisc")
if(!require(data.table)) install.packages("data.table")

# Import libraries
library(plyr)
library(dplyr)
library(caret)
library(tinytex)
library(funModeling)
library(stringr)
library(lubridate)
library(ggplot2)
library(ggthemes)
library(mltools)
library(Hmisc)
library(data.table)

# The following lines of code are provided by the instructor in Professional Certificate in Data Science - Rafael Irizarry. They construct the train and valition MovieLens datasets needed for the analysis

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove unneeded objects
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Create feature engineering on the modelling sample
edx_features = edx %>%
  mutate(release_year = as.numeric(str_sub(title,-5,-2)) # extract all between the last 5 characters and the last 2 characters in the title column and convert to numeric - this is the year the movie was released
         , title = str_sub(title,-1*(length(title)),-7) # extract all from the beginning of the character to the last 7 characters in the title column
         , Index_Genre_Delimiter = gregexpr("\\|",genres)
  ) %>%
  rowwise() %>%
  mutate(Index_First_Genre_Delimiter = Index_Genre_Delimiter[1]) %>%
  ungroup() %>%
  mutate(leading_genre = case_when(Index_First_Genre_Delimiter != -1 ~ str_sub(genres,1,Index_First_Genre_Delimiter-1)
                                   , TRUE ~ genres
  )
  ) %>%
  select(-Index_Genre_Delimiter,-Index_First_Genre_Delimiter) %>%
  select(-title) %>% # remove the title column as it is not needed anymore
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(YRG = as.numeric(year(timestamp))) %>%
  rowwise() %>% #perform the operation row wise
  mutate(YAR = 2009-release_year
         , YRG_AR = YRG - release_year
  ) %>%
  ungroup()  %>% # ungroup after the rowwise function was used
  group_by(movieId) %>%
  mutate(NR = n()) %>%
  ungroup() 

### Create Model ###
mu <- mean(edx_features$rating) 
# Estimate movieId effect
movie_avgs <- edx_features %>% 
  group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu)) %>%
  ungroup()

# Estimate userId efect
user_avgs <- edx_features %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

# Estimate year after release effect
yar_avgs <- edx_features %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(YAR) %>%
  summarise(b_yar = mean(rating - mu - b_i - b_u))

# Estimate genre effect
genre_avgs <- edx_features %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yar_avgs, by='YAR') %>%
  group_by(genres) %>%
  summarise(b_gen = mean(rating - mu - b_i - b_u - b_yar))

# Create feature engineering on validation set
validation_features = validation %>%
  mutate(release_year = as.numeric(str_sub(title,-5,-2)) # extract all between the last 5 characters and the last 2 characters in the title column and convert to numeric - this is the year the movie was released
         , title = str_sub(title,-1*(length(title)),-7) # extract all from the beginning of the character to the last 7 characters in the title column
         , Index_Genre_Delimiter = gregexpr("\\|",genres)
  ) %>%
  rowwise() %>%
  mutate(Index_First_Genre_Delimiter = Index_Genre_Delimiter[1]) %>%
  ungroup() %>%
  mutate(leading_genre = case_when(Index_First_Genre_Delimiter != -1 ~ str_sub(genres,1,Index_First_Genre_Delimiter-1)
                                   , TRUE ~ genres
  )
  ) %>%
  select(-Index_Genre_Delimiter,-Index_First_Genre_Delimiter) %>%
  select(-title) %>% # remove the title column as it is not needed anymore
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(YRG = as.numeric(year(timestamp))) %>%
  rowwise() %>% #perform the operation row wise
  mutate(YAR = 2009-release_year
         , YRG_AR = YRG - release_year
  ) %>%
  ungroup()  %>% # ungroup after the rowwise function was used
  group_by(movieId) %>%
  mutate(NR = n()) %>%
  ungroup() 

# Make predictions and estimate RMSE on validation set
predicted_ratings <- validation_features %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yar_avgs, by='YAR') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_yar + b_gen) %>%
  .$pred

# Calculate RMSE
model_rmse <- RMSE(predicted_ratings, validation_features$rating)

# Print rmse to the console
model_rmse

















