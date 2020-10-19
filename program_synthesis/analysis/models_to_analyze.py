model_labels = """
vanilla
#vanilla-123-old
#vanilla-123-old-real-nearai-finetuned-1e-4
#vanilla-123-old-real-nearai32-finetuned-1e-4
#vanilla-123-old-finetuned-agg-start-with-finetuned
#vanilla-1-old
#vanilla-1-old-real-nearai-finetuned-1e-4

aggregate-with-io
# aggregate-with-io-123-old
# aggregate-with-io-123-old-real-nearai-finetuned-1e-4
# aggregate-with-io-123-old-real-nearai32-finetuned-1e-4

vanilla-real-nearai-finetuned-1e-4
aggregate-with-io-real-nearai-finetuned-1e-4

# vanilla-real-nearai-finetuned-rl-1e-5
# aggregate-with-io-real-nearai-finetuned-rl-1e-5

vanilla-many-mutations
aggregate-with-io-many-mutations
# vanilla-real-nearai-finetuned-rl-1e-5-use-heldout
# aggregate-with-io-real-nearai-finetuned-rl-1e-5-use-heldout

# vanilla-real-nearai-finetuned-1e-5
# aggregate-with-io-real-nearai-finetuned-1e-5

vanilla-real-nearai32-finetuned-1e-4
aggregate-with-io-real-nearai32-finetuned-1e-4

vanilla,h,327
vanilla,h,327,finetuned-nearai-1e-4
vanilla,h,327,finetuned-nearai32-1e-4
""".strip().split("\n")
model_labels = [x for x in model_labels if x and x[0] != '#']