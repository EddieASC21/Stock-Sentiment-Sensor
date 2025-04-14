import os
import pandas as pd

VOTES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "post_votes.csv")

def load_votes():
    try:
        votes_df = pd.read_csv(VOTES_FILE)
        votes = {}
        for _, row in votes_df.iterrows():
            votes[str(row['post_id'])] = (row['upvotes'], row['downvotes'])
        return votes
    except FileNotFoundError:
        return {}

def save_votes(votes):
    data = []
    for post_id, (up, down) in votes.items():
        data.append({'post_id': post_id, 'upvotes': up, 'downvotes': down})
    df = pd.DataFrame(data)
    df.to_csv(VOTES_FILE, index=False)

def update_vote(post_id, vote):
    votes = load_votes()
    post_id = str(post_id)
    current = votes.get(post_id, (0, 0))
    if vote == "up":
        votes[post_id] = (current[0] + 1, current[1])
    elif vote == "down":
        votes[post_id] = (current[0], current[1] + 1)
    save_votes(votes)
    return votes[post_id]

def get_vote_counts(post_id):
    votes = load_votes()
    return votes.get(str(post_id), (0, 0))
