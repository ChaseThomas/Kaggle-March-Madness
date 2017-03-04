sigmoid <- function (x) {
  return(1.0 / (1.0 + exp(-1.0 * x)))
}

submission <- read.csv("sample_submission.csv")
submission <- tidyr::separate(submission, id, c("Season", "FirstTeam", "SecondTeam"),
                              sep = "_")

submission$WTeam <- submission$FirstTeam

# logistic regression model based on elo rating, k = 0.01
for (i in c(1:nrow(submission))) {
  id_1 <- submission[i,]$FirstTeam
  id_2 <- submission[i,]$SecondTeam
  
  elo_1 <- get_elo_of(id_1)
  elo_2 <- get_elo_of(id_2)
  
  elo_diff <- elo_1 - elo_2
  
  submission[i,]$WTeam <- if (elo_1 > elo_2) id_1 else id_2
  submission[i,]$pred <- sigmoid(0.01 * abs(elo_diff))
}

get_wfgm_of <- function(id) {
  return(team[which(team$Team_Id == id),]$Wfgm)
}

write.csv(submission, "submission.csv", row.names = FALSE)


