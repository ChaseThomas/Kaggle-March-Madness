library(aod)
library(ggplot2)
library(Rcpp)

setwd("Desktop/Programming/Kaggle/March Madness/")
mydata <- read.csv("TourneyDetailedResults.csv")

summary(mydata)

mydata <- mydata[-c(7)]
sapply(mydata, mean)

team <- read.csv("Teams.csv")

mydata$WfgPer <- mydata$Wfgm / mydata$Wfga
mydata$Wfg3Per <- mydata$Wfgm3 / mydata$Wfga3
mydata$WftPer <- mydata$Wftm / mydata$Wfta
mydata$Woff <- mydata$Wfga + mydata$Wfga3 + mydata$Wfta +
                  mydata$Wor + mydata$Wast + mydata$Wto + mydata$Wstl
mydata$Wdef <- mydata$Wdr + mydata$Wblk + mydata$Wto + mydata$Wstl

WfgPer <- aggregate(formula = WfgPer~Wteam,  #aggregate mean
                  data = mydata, FUN=mean)
Wfg3Per <- aggregate(formula = Wfg3Per~Wteam,  
                    data = mydata, FUN=mean)
WftPer <- aggregate(formula = WftPer~Wteam, 
                    data = mydata, FUN=mean)
Woff <- aggregate(formula = Woff~Wteam, 
                    data = mydata, FUN=mean)
Wdef <- aggregate(formula = Wdef~Wteam, 
                    data = mydata, FUN=mean)


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

for (i in c(1:nrow(mydata))) {
  w_id <- mydata[i,]$Wteam
  l_id <- mydata[i,]$Lteam
  
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
