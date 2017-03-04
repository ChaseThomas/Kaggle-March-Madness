library(rpart)
library(rpart.plot)

substat <- submission

for (i in c(1:nrow(substat))) {
  id_1 <- submission[i,]$FirstTeam
  id_2 <- submission[i,]$SecondTeam
  
  substat$WfgDiff[i] <- get_wfgper_of(id_1) - get_wfgper_of(id_2)
  substat$Wfg3Diff[i] <- get_wfg3per_of(id_1) - get_wfg3per_of(id_2)
  substat$WftDiff[i] <- get_wftper_of(id_1) - get_wftper_of(id_2)
  substat$WoffDiff[i] <- get_woffper_of(id_1) - get_woffper_of(id_2)
  substat$WdefDiff[i] <- get_wdefper_of(id_1) - get_wdefper_of(id_2)
}

get_wfgper_of <- function(id) {
  return(team[which(team$Team_Id == id),]$WfgPer)
}

get_wfg3per_of <- function(id) {
  return(team[which(team$Team_Id == id),]$Wfg3Per)
}

get_wftper_of <- function(id) {
  return(team[which(team$Team_Id == id),]$WftPer)
}

get_woffper_of <- function(id) {
  return(team[which(team$Team_Id == id),]$WoffPer)
}

get_wdefper_of <- function(id) {
  return(team[which(team$Team_Id == id),]$WdefPer)
}

dt = rpart(WTeam ~ WfgDiff + Wfg3Diff + WftDiff + WoffDiff +
                         WdefDiff, data = substat, method="class", minbucket=25)

prp(dt)

#PredictCART = predict(dt, newdata = test, type = "class")