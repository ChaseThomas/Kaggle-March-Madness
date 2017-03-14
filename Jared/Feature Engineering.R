library(aod)
library(ggplot2)
library(Rcpp)

setwd("/Users/jaredlim/Desktop/Programming/Kaggle/Kaggle-March-Madness/Jared/")
tourney <- read.csv("TourneyDetailedResults.csv")
regular <- read.csv("RegularSeasonDetailedResults.csv")
future_prediction <- read.csv("future_predictions.csv")
historical_prediction <- read.csv("historical_predictions.csv")

data <- tourney[which(tourney$Season > 2013),]
data <- rbind(tourney, regular[which(regular$Season > 2013),])

summary(data)

data <- data[-c(7)]
sapply(data, mean)

team <- read.csv("Teams.csv")

data$WfgPer <- data$Wfgm / data$Wfga
data$Wfg3Per <- data$Wfgm3 / data$Wfga3
data$WftPer <- data$Wftm / data$Wfta
data$Woff <- data$Wfga + data$Wfga3 + data$Wfta +
                  data$Wor + data$Wast + data$Wto + data$Wstl
data$Wdef <- data$Wdr + data$Wblk + data$Wto + data$Wstl

WfgPer <- aggregate(formula = WfgPer~Wteam,  #aggregate mean
                  data = data, FUN=mean)
Wfg3Per <- aggregate(formula = Wfg3Per~Wteam,  
                    data = data, FUN=mean)
WftPer <- aggregate(formula = WftPer~Wteam, 
                    data = data, FUN=mean)
Woff <- aggregate(formula = Woff~Wteam, 
                    data = data, FUN=mean)
Wdef <- aggregate(formula = Wdef~Wteam, 
                    data = data, FUN=mean)


colnames(WfgPer)[1] = "Team_Id"
colnames(Wfg3Per)[1] = "Team_Id"
colnames(WftPer)[1] = "Team_Id"
colnames(Woff)[1] = "Team_Id"
colnames(Wdef)[1] = "Team_Id"

team <- plyr::join(team, WfgPer,
                      by='Team_Id',type='left')
team <- plyr::join(team, Wfg3Per,
                   by='Team_Id',type='left')
team <- plyr::join(team, WftPer,
                   by='Team_Id',type='left')
team <- plyr::join(team, Woff,
                   by='Team_Id',type='left')
team <- plyr::join(team, Wdef,
                   by='Team_Id',type='left')

get_elo_of <- function(id) {
  return(team[which(team$Team_Id == id),]$Elo)
}

# Elo rating
team$Elo <- 1500

for (i in c(1:nrow(data))) {
  w_id <- data[i,]$Wteam
  l_id <- data[i,]$Lteam
  
  w_rating <- get_elo_of(w_id)
  l_rating <- get_elo_of(l_id)
  
  w_expected <- 1.0/(1.0+10**(l_rating - w_rating)/400.0)
  l_expected <- 1.0/(1.0+10**(w_rating - l_rating)/400.0)
  
  # winning = 1, losing = 0, K= 20
  w_rating <- plyr::round_any(w_rating + 15 * l_expected, 1)
  l_rating <- plyr::round_any(l_rating + 15 * w_expected, 1)
  
  team[which(team$Team_Id == w_id),]$Elo = w_rating
  team[which(team$Team_Id == l_id),]$Elo = l_rating
}
  