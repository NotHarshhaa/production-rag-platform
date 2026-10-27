# Evaluating RAG Systems

## Retrieval quality

Retrieval is measured with recall at k, precision at k, and MRR. Recall at k
answers how many of the known relevant documents were found in the top k
results. Precision at k penalizes noise among the top results. MRR rewards
systems that put the best document first.

## Answer quality

Faithfulness checks that every statement in the answer is supported by the
retrieved context. Answer relevance checks that the answer actually addresses
the question that was asked. Citation precision verifies that cited chunks
were part of the retrieved evidence.

## Baselines and regression

Before changing chunking, embeddings, retrieval algorithms, or prompts, run
the evaluation suite and record a baseline. After the change, run the same
suite and compare. A change that improves one metric while regressing
another should be reviewed carefully.

Automated scores are signals, not proof. Important answers must still be
validated against human-reviewed examples.
