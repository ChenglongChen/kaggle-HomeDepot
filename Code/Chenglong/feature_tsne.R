# 
# @author: Chenglong Chen <c.chenglong@gmail.com>
# @brief: tsne based features
# 

require(data.table)
require(Rtsne)

# random seed for reproducibility
set.seed(2016)

# path
setwd(".")
feat_dir <- "../../Feat/"

# feature names
fnames <- c(
    "LSA100_Word_Unigram_Pair_search_term_x_product_title_100D",
    "LSA100_Word_Bigram_Pair_search_term_x_product_title_100D",
    "LSA100_Word_Obs_Unigram_Target_Unigram_Cooc_search_term_x_product_title_100D",
    "LSA100_Word_Obs_Unigram_Target_Bigram_Cooc_search_term_x_product_title_100D"
)

# setting
perplexity <- 30
theta <- 0.5
dims <- 2

# run
for(fname in fnames) {
    # load lsa features
    file_lsa <- paste(feat_dir, fname, ".csv", sep="")
    X <- fread(file_lsa, data.table=F)
    X <- as.matrix(X)
    gc()

    # run tsne
    tsne <- Rtsne(X , check_duplicates=FALSE, pca=FALSE,
                  perplexity=perplexity, theta=theta, dims=dims)

    # save tsne features
    col.names <- paste("TSNE_", 1:ncol(tsne$Y), sep="")
    file_tsne <- paste(feat_dir, "/TSNE_", fname, ".csv", sep="")
    write.table(tsne$Y, file=file_tsne, sep=',', quote=FALSE, 
                row.names=FALSE, col.names=col.names)
}
