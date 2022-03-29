library(scales)
library(ggplot2)
library(reshape2)
library(wesanderson)

res <- read.csv('res.csv')
res <- as.data.frame(t(res))
header.true <- function(df) {
  names(df) <- as.character(unlist(df[1,]))
  df[-1,]
}
res<-header.true(res)
res <- cbind(method = rownames(res), res)
rownames(res) <- 1:nrow(res)
res$Darmanis <- as.numeric(res$Darmanis)
# 
# ggplot(res,aes(x=method, y=Darmanis)) + geom_bar(stat="identity")+
#   ylim(0,1) + theme_bw()


res2<- melt(res, id.vars ="method")
res2$value <- as.numeric(res2$value)

ggplot(res2, aes(x=method, y=value, fill=method))+
  geom_bar(stat="identity", colour="black")+
  facet_wrap(~variable)+theme_bw()+
  theme(legend.position="none",strip.background = element_rect(
    color="black", fill="#FFFFFF", linetype="solid"),strip.text.x = element_text(size = 15),
    axis.text.x = element_text(angle = 45, hjust=1, size=15)) +
  scale_fill_manual(values=wes_palette(n=4, name="GrandBudapest2")) + 
  scale_y_continuous(expand = c(0,0), limits = c(0, 1))+xlab("Methods")+
  ylab("ARI")
