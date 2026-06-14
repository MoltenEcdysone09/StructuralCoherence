# install.packages("signnet")
library(igraph)
library(signnet)

g <- graph_from_edgelist(matrix(c(1,2, 2,3, 3,1), ncol=2), directed=TRUE)

E(g)$sign <- c(1, -1, 1)

census <- triad_census_signed(g)

signnet_codes <- names(census)

write.csv(signnet_codes, "signnet_138_codes.csv", row.names=FALSE)
