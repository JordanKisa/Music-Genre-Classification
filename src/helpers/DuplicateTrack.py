import os
import difflib

def find_similar_names(directory, similarity_threshold):
    similar_pairs = []

    for root, _, files in os.walk(directory):
        for i, filename1 in enumerate(files):
            for filename2 in files[i+1:]:
                similarity = difflib.SequenceMatcher(None, filename1, filename2).ratio()
                if similarity > similarity_threshold:
                    similar_pairs.append(( similarity, os.path.join(root, filename1), os.path.join(root, filename2)))

    return similar_pairs

if __name__ == "__main__":
    target_directory = "../../data"
    similarity_threshold = 0.80  # Adjust as needed

    similar_pairs = find_similar_names(target_directory, similarity_threshold)

    if similar_pairs:
        print("Similar File Name Pairs:")
        similar_pairs.sort(key=lambda x: x[0])
        for val, file1, file2 in similar_pairs:
            print(f"Similarity: {round(val,2)} | {file1} | {file2} \n")
    else:
        print("No similar file name pairs found.")
