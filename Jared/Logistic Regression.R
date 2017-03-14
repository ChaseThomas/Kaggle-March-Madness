sigmoid <- function (x) {
  return(1.0 / (1.0 + exp(-1.0 * x)))
}

#submission <- read.csv("sample_submission.csv")
submission <- historical_prediction
submission <- future_prediction
#submission <- tidyr::separate(submission, id, c("Season", "FirstTeam", "SecondTeam"), sep = "_")

#submission$WTeam <- submission$FirstTeam

# logistic regression model based on elo rating, k = 0.01
for (i in c(1:nrow(submission))) {
  id_1 <- submission[i,]$team1
  id_2 <- submission[i,]$team2
  
  elo_1 <- get_elo_of(id_1)
  elo_2 <- get_elo_of(id_2)
  
  if (elo_1 == 1500) elo_1 <- elo_2
  if (elo_2 == 1500) elo_2 <- elo_1
  
  submission$elo_diff[i] <- get_elo_of(id_1) - get_elo_of(id_2)
  submission$wfg_diff[i] <- get_wfg_of(id_1) - get_wfg_of(id_2)
  submission$wfg3_diff[i] <- get_wfg3_of(id_1) - get_wfg3_of(id_2)
  submission$wft_diff[i] <- get_wft_of(id_1) - get_wft_of(id_2)
  submission$woff_diff[i] <- get_woff_of(id_1) - get_woff_of(id_2)
  submission$wdef_diff[i] <- get_wdef_of(id_1) - get_wdef_of(id_2)
  
  #submission[i,]$WTeam <- if (elo_1 > elo_2) id_1 else id_2
  #submission[i,]$predicted <- sigmoid(0.01 * elo_diff)
}

get_elo_of <- function(id) {
  return(team[which(team$Team_Id == id),]$Elo)
}

get_wfg_of <- function(id) {
  return(team[which(team$Team_Id == id),]$WfgPer)
}

get_wfg3_of <- function(id) {
  return(team[which(team$Team_Id == id),]$Wfg3Per)
}

get_wft_of <- function(id) {
  return(team[which(team$Team_Id == id),]$WftPer)
}

get_woff_of <- function(id) {
  return(team[which(team$Team_Id == id),]$Woff)
}

get_wdef_of <- function(id) {
  return(team[which(team$Team_Id == id),]$Wdef)
}

logit <- glm(actual ~ elo_diff + wfg_diff +
               wfg3_diff + wft_diff +
               woff_diff + wdef_diff, data = submission,
             family = binomial())


write.csv(submission[,c(1:3)], "future_submissions.csv", row.names = FALSE)

submission$predicted <- predicted

predicted <- predict(logit, type="response")
predicted <- predict(logit, newdata = submission, type="response")
predicted <- ifelse(predicted > 0.5,1,0)

misClasificError <- mean(predicted != submission$actual)
print(paste('Accuracy',1-misClasificError))
