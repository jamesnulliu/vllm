from vllm.sequence import SampleLogprobs

import pandas as pd

def logprobs_to_df(sample_logprobs: SampleLogprobs) -> pd.DataFrame:
    """
    Converts a list of logprob dictionaries into a pandas DataFrame.

    Each item in the list represents a token generation step.
    Each dictionary contains the top-N logprobs for that step.

    Args:
        sample_logprobs: A list of dictionaries, where each dictionary maps
                         a token ID to a Logprob object.

    Returns:
        A pandas DataFrame with a multi-level column index, organizing
        logprobs by rank.
    """
    processed_data = []
    if not sample_logprobs:
        return pd.DataFrame()

    # Determine the maximum number of logprobs (ranks) across all steps
    # to ensure consistent column structure.
    max_ranks = 0
    for token_logprobs in sample_logprobs:
        if token_logprobs:
            max_ranks = max(
                max_ranks, max(
                    lp.rank 
                    for lp in token_logprobs.values() 
                    if lp.rank is not None
                )
            )

    # Iterate over each token's logprobs in the list
    for token_logprobs in sample_logprobs:
        # Sort the logprobs by rank to ensure correct column order
        # The key of the dict is the token_id, the value is the Logprob object
        sorted_logprobs = sorted(
            token_logprobs.items(), 
            key=lambda item: item[1].rank or float('inf')
        )

        row_data = {}
        for token_id, logprob_info in sorted_logprobs:
            rank = logprob_info.rank
            if rank is not None:
                # Use rank to create unique column names for each piece of info
                row_data[f'tokenid_{rank}'] = token_id
                row_data[f'logprob_{rank}'] = logprob_info.logprob
                row_data[f'decoded_{rank}'] = logprob_info.decoded_token
        processed_data.append(row_data)

    # Create the initial DataFrame
    df = pd.DataFrame(processed_data)


    return df