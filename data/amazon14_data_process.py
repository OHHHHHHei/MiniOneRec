import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import datetime
import torch
from tqdm import tqdm
import numpy as np


def clean_text(text):
    """Clean text by removing HTML tags and excessive whitespace"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Decode HTML entities
    text = html.unescape(text)
    # Replace quotes
    text = text.replace("&quot;", "\"").replace("&amp;", "&")
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def check_path(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def write_json_file(data, file_path):
    """Write data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def write_remap_index(index_map, file_path):
    """Write index mapping to file"""
    with open(file_path, 'w') as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")


def get_timestamp_start(year, month):
    """Get timestamp for the start of a given year and month"""
    return int(datetime.datetime(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())


def open_file(file_path):
    """Open file handling both raw json and gzipped json"""
    if file_path.endswith('.gz'):
        return gzip.open(file_path, 'rt', encoding='utf-8')
    else:
        return open(file_path, 'r', encoding='utf-8')


def load_metadata_json(category, metadata_file=None):
    """Load metadata"""
    if metadata_file is None:
        print(f"Error: Metadata file must be provided")
        return [], {}, set()
    
    metadata = []
    try:
        # Check if file is gzipped or raw json
        with open_file(metadata_file) as f:
            for line in f:
                try:
                    # Try standard JSON first
                    meta = json.loads(line)
                    metadata.append(meta)
                except json.JSONDecodeError:
                    try:
                        # Fallback to eval for single-quoted Python dicts (common in 2014 data)
                        meta = eval(line)
                        metadata.append(meta)
                    except:
                        continue
    except FileNotFoundError:
        print(f"Metadata file {metadata_file} not found")
        return [], {}, set()
    
    id_title = {}
    remove_items = set()
    
    for meta in tqdm(metadata, desc="Processing metadata"):
        # 2014 data usually has 'asin'
        if 'asin' not in meta:
            continue
            
        asin = meta['asin']
        
        # Check title
        if 'title' not in meta:
            remove_items.add(asin)
            continue
            
        title = meta['title']
        if not title:
            remove_items.add(asin)
            continue
            
        # Clean title
        title = clean_text(title)
        meta['title'] = title # Update in place
        
        if len(title) > 0:
            id_title[asin] = title
        else:
            remove_items.add(asin)
    
    return metadata, id_title, remove_items


def load_reviews_json(category, reviews_file=None):
    """Load reviews"""
    if reviews_file is None:
        print(f"Error: Reviews file must be provided")
        return []
    
    reviews = []
    try:
        with open_file(reviews_file) as f:
            for line in f:
                try:
                    review = json.loads(line)
                    reviews.append(review)
                except json.JSONDecodeError:
                    try:
                        review = eval(line)
                        reviews.append(review)
                    except:
                        continue
    except FileNotFoundError:
        print(f"Reviews file {reviews_file} not found")
        return []
    
    return reviews


def k_core_filtering(reviews, id_title, K=5, start_timestamp=None, end_timestamp=None):
    """Perform k-core filtering"""
    remove_users = set()
    remove_items = set()
    
    # Remove items without titles
    for review in reviews:
        if review['asin'] not in id_title:
            remove_items.add(review['asin'])
    
    # Iterative k-core filtering
    loop_count = 0
    while True:
        loop_count += 1
        new_reviews = []
        flag = False
        total = 0
        user_counts = dict()
        item_counts = dict()
        
        for review in tqdm(reviews, desc=f"K-core filtering (Batch {loop_count})"):
            # Filter by timestamp
            if start_timestamp and end_timestamp:
                if 'unixReviewTime' not in review:
                    continue
                if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
                    continue
            
            if review['reviewerID'] in remove_users or review['asin'] in remove_items:
                continue
            
            if review['reviewerID'] not in user_counts:
                user_counts[review['reviewerID']] = 0
            user_counts[review['reviewerID']] += 1
            
            if review['asin'] not in item_counts:
                item_counts[review['asin']] = 0
            item_counts[review['asin']] += 1
            
            total += 1
            new_reviews.append(review)
        
        # Mark users/items for removal if below threshold
        # For the first loop, we might want to be lenient, but standard K-core applies rigorous threshold
        
        current_remove_users = 0
        current_remove_items = 0
        
        for user in user_counts:
            if user_counts[user] < K:
                remove_users.add(user)
                current_remove_users += 1
                flag = True
        
        for item in item_counts:
            if item_counts[item] < K:
                remove_items.add(item)
                current_remove_items += 1
                flag = True
        
        print(f"Batch {loop_count}: Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {total}")
        print(f"  Removed in this batch - Users: {current_remove_users}, Items: {current_remove_items}")
        
        if not flag:
            break
        
        reviews = new_reviews
    
    return new_reviews, user_counts, item_counts


def convert_inters2dict(reviews):
    """Convert interactions to dict format"""
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    
    # Sort reviews by timestamp for each user
    user_reviews = collections.defaultdict(list)
    for review in reviews:
        user_reviews[review['reviewerID']].append(review)
    
    # Sort each user's reviews by timestamp
    for user in user_reviews:
        user_reviews[user].sort(key=lambda x: int(x['unixReviewTime']))
    
    # Create mappings and interactions
    interactions = []
    for user in user_reviews:
        if user not in user2index:
            user2index[user] = len(user2index)
        
        user_items = []
        for review in user_reviews[user]:
            item = review['asin']
            if item not in item2index:
                item2index[item] = len(item2index)
            
            user_items.append(item)
            # user_id, item_id, rating, timestamp
            interactions.append((
                user, item, 
                float(review.get('overall', 0.0)), 
                int(review['unixReviewTime'])
            ))
        
        user2items[user2index[user]] = [item2index[item] for item in user_items]
    
    return user2items, user2index, item2index, interactions


def generate_interaction_list(reviews, user2index, item2index, id_title):
    """Generate interaction list for splitting"""
    # Create user interactions
    interact = dict()
    item2id = {item: idx for item, idx in item2index.items()}
    
    for review in tqdm(reviews, desc="Building interaction list"):
        user = review['reviewerID']
        item = review['asin']
        
        if user not in interact:
            interact[user] = {
                'items': [],
                'ratings': [],
                'timestamps': [],
                'item_ids': [],
                'titles': []
            }
        
        interact[user]['items'].append(item)
        interact[user]['ratings'].append(review.get('overall', 0.0))
        interact[user]['timestamps'].append(review['unixReviewTime'])
        interact[user]['item_ids'].append(item2id[item])
        interact[user]['titles'].append(id_title[item])
    
    # Sort by timestamp for each user
    interaction_list = []
    for user in tqdm(interact.keys(), desc="Creating interaction sequences"):
        items = interact[user]['items']
        ratings = interact[user]['ratings']
        timestamps = interact[user]['timestamps']
        item_ids = interact[user]['item_ids']
        titles = interact[user]['titles']
        
        # Sort all by timestamp
        all_data = list(zip(items, ratings, timestamps, item_ids, titles))
        all_data.sort(key=lambda x: int(x[2]))
        items, ratings, timestamps, item_ids, titles = zip(*all_data)
        items, ratings, timestamps, item_ids, titles = list(items), list(ratings), list(timestamps), list(item_ids), list(titles)
        
        # Create sequences (sliding window with max history of 10)
        # Consistent with MiniOneRec requirement
        for i in range(1, len(items)):
            st = max(i - 10, 0)
            interaction_list.append([
                user,                    # user_id
                items[st:i],            # item_asins (history)
                items[i],               # item_asin (target)
                item_ids[st:i],         # history_item_id
                item_ids[i],            # item_id (target)
                titles[st:i],           # history_item_title
                titles[i],              # item_title (target)
                ratings[st:i],          # history_rating
                ratings[i],             # rating (target)
                timestamps[st:i],       # history_timestamp
                timestamps[i]           # timestamp (target)
            ])
    
    # Sort by timestamp for chronological split
    interaction_list.sort(key=lambda x: int(x[-1]))
    return interaction_list


def convert_to_atomic_files(args, interaction_list, user2index):
    """Convert interaction list to train/valid/test files using 8:1:1 split"""
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    
    # Create output directories
    check_path(os.path.join(args.output_path, args.dataset))
    
    # Split 8:1:1
    total_len = len(interaction_list)
    train_end = int(total_len * 0.8)
    valid_end = int(total_len * 0.9)
    
    train_interactions = interaction_list[:train_end]
    valid_interactions = interaction_list[train_end:valid_end]
    test_interactions = interaction_list[valid_end:]
    
    print(f"Train interactions: {len(train_interactions)}")
    print(f"Valid interactions: {len(valid_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")
    
    def write_file(filename, interactions):
        with open(os.path.join(args.output_path, args.dataset, filename), 'w') as file:
            file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for interaction in interactions:
                user_id_original = interaction[0]
                user_id = user2index[user_id_original]  
                history_item_ids = [str(x) for x in interaction[3]]  # history item ids
                target_item_id = str(interaction[4])  # target item id
                
                # Limit history to last 50 items
                history_seq = history_item_ids[-50:]
                file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')

    write_file(f'{args.dataset}.train.inter', train_interactions)
    write_file(f'{args.dataset}.valid.inter', valid_interactions)
    write_file(f'{args.dataset}.test.inter', test_interactions)
    
    return train_interactions, valid_interactions, test_interactions


def load_review_data(reviews, user2index, item2index):
    """Load review data"""
    review_data = {}
    
    for review in tqdm(reviews, desc='Load reviews text'):
        try:
            user = review['reviewerID']
            item = review['asin']
            
            if user in user2index and item in item2index:
                uid = user2index[user]
                iid = item2index[item]
                
                # Use timestamp to create unique keys
                timestamp = review['unixReviewTime']
                unique_key = str((uid, iid, timestamp))
                
            else:
                continue
                
            review_text = clean_text(review.get('reviewText', ''))
            summary = clean_text(review.get('summary', ''))
                
            review_data[unique_key] = {"review": review_text, "summary": summary}
            
        except (ValueError, KeyError):
            continue
    
    return review_data


def create_item_features(metadata, item2index, id_title):
    """Create item features"""
    item2feature = collections.defaultdict(dict)
    
    # Create a mapping from asin to metadata
    asin_to_meta = {}
    for meta in metadata:
        asin_to_meta[meta['asin']] = meta
    
    for item_asin, item_id in item2index.items():
        if item_asin in asin_to_meta:
            meta = asin_to_meta[item_asin]
            
            title = id_title.get(item_asin, clean_text(meta.get("title", "")))
            
            descriptions = meta.get("description", "")
            if descriptions:
                descriptions = clean_text(descriptions)
            else:
                descriptions = ""
                
            brand = meta.get("brand", "").replace("by\n", "").strip()
            
            # Handle categories (sometimes list of lists, sometimes string)
            categories = meta.get("categories", [])
            # In 2014 data, categories are often [[Cat1, Cat2, Cat3], [Cat1, Cat4]]
            if categories and len(categories) > 0:
                # Flatten
                flat_cats = []
                for cat_path in categories:
                    if isinstance(cat_path, list):
                        flat_cats.extend(cat_path)
                    else:
                        flat_cats.append(cat_path)
                # Deduplicate and join
                categories = ",".join(list(set(flat_cats))).strip()
            else:
                categories = ""
            
            item2feature[item_id] = {
                "title": title,
                "description": descriptions,
                "brand": brand,
                "categories": categories
            }
    
    return item2feature


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='Dataset name')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--st_year', type=int, default=1996, help='start year')
    parser.add_argument('--st_month', type=int, default=1, help='start month')
    parser.add_argument('--ed_year', type=int, default=2014, help='end year')
    parser.add_argument('--ed_month', type=int, default=12, help='end month')
    parser.add_argument('--metadata_file', type=str, required=True, help='metadata file path (json or json.gz)')
    parser.add_argument('--reviews_file', type=str, required=True, help='reviews file path (json or json.gz)')
    parser.add_argument('--output_path', type=str, default='./data', help='output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print(f'Processing dataset: {args.dataset}')
    print(f'Time range: {args.st_year}-{args.st_month} to {args.ed_year}-{args.ed_month}')
    print(f'K-core threshold: {args.user_k}')
    
    # Set time range
    start_timestamp = get_timestamp_start(args.st_year, args.st_month)
    end_timestamp = get_timestamp_start(args.ed_year, args.ed_month)
    
    # Load metadata first
    print("Loading metadata...")
    metadata, id_title, remove_items = load_metadata_json(
        args.dataset, args.metadata_file
    )
    
    if not metadata:
        print("Failed to load metadata.")
        exit(1)
        
    print(f"Loaded {len(metadata)} metadata items. ({len(id_title)} valid titles)")

    # Load reviews
    print("Loading reviews...")
    reviews = load_reviews_json(
        args.dataset, args.reviews_file
    )
    
    if not reviews:
        print(f"Error: No reviews found.")
        exit(1)
        
    print(f"Loaded {len(reviews)} total reviews")
    
    # Core filtering
    filtered_reviews, user_counts, item_counts = k_core_filtering(
        reviews, id_title, args.user_k, start_timestamp, end_timestamp
    )
    
    print(f"Final filtering results:")
    print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {len(filtered_reviews)}")
    
    if len(filtered_reviews) == 0:
        print("No reviews left after filtering. Check your time range or K-core settings.")
        exit(1)

    # Convert to standard format
    print("Converting to standard inter format...")
    user2items, user2index, item2index, interactions = convert_inters2dict(filtered_reviews)
    
    # Generate interaction list for 8:1:1 split
    print("Generating interaction list for 8:1:1 split...")
    interaction_list = generate_interaction_list(
        filtered_reviews, user2index, item2index, id_title
    )
    
    # Create output directory and split data
    train_interactions, valid_interactions, test_interactions = convert_to_atomic_files(
        args, interaction_list, user2index
    )
    
    # Generate user2items for compatibility
    user2items_final = collections.defaultdict(list)
    for user_idx, item_list in user2items.items():
        user2items_final[user_idx] = item_list
    
    # Write output files
    output_dir = os.path.join(args.output_path, args.dataset)
    write_json_file(user2items_final, os.path.join(output_dir, f'{args.dataset}.inter.json'))
    
    # Create item features
    print("Creating item features...")
    item2feature = create_item_features(metadata, item2index, id_title)
    
    # Load review data
    print("Loading review data details...")
    review_data = load_review_data(filtered_reviews, user2index, item2index)
    
    print(f"Writing final JSON files...")
    write_json_file(item2feature, os.path.join(output_dir, f'{args.dataset}.item.json'))
    write_json_file(review_data, os.path.join(output_dir, f'{args.dataset}.review.json'))
    
    write_remap_index(user2index, os.path.join(output_dir, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(output_dir, f'{args.dataset}.item2id'))
    
    print("Processing completed!")
