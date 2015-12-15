# QANTA Reloaded
*A reimplementation of the QANTA algorithm*

## About
This repository contains a university project. We aimed to reimplement the QANTA algorithm developed by Iyyer et al. (2014), as we found it very difficult to extend the [existing implementation](https://www.cs.umd.edu/~miyyer/qblearn/) to new data.

As it stands, this reimplementation appears to have inferior prediction performance to the original, but as the original authors cannot share most of their data, we cannot know for sure. See below for known differences between this and the original implementation.

QANTA is short for ``question answering neural network with trans-sentential averaging''.

The preprocessing step relies on the Stanford Parser, version 3.5.2.

## Known main differences to the original implementation
- This implementation uses cosine distance for prediction instead of a logistic regression classifier. This is because we (I) did not see the connection between the paper and the original implementation's way of predicting.
- This implementation does not use stop-words removal in the evaluation. This is because the paper does not describe the stopword removal.
- This implementation initializes parameters uniformly in the range [-1,1], rather than in the hardcoded and unexplained range [-0.173,0.173] from the original implementation.


## References
- Iyyer, M., Boyd-Graber, J., Claudino, L., Socher, R., & Daumé III, H. (2014). A neural network for factoid question answering over paragraphs. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 633-644).
