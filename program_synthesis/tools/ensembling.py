from models.base import InferenceResult


def ensembled_inference(inference_functions, mode):
    """
    Ensemble the given inference functions together via dovetailing their results.
    """

    if mode == "none":
        assert len(inference_functions) == 1
        return inference_functions[0]

    def inference(batch):
        results_each = [f(batch) for f in inference_functions]
        assert mode == "dovetail", "no other modes supported as of yet"
        assert len({len(r) for r in results_each}) == 1
        return [
            InferenceResult.dovetail([r[i] for r in results_each])
            for i in range(len(results_each[0]))
        ]

    return inference
