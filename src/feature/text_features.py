def get_text_features(dataset):
    features = []
    for document in dataset.documents:
        fv = {}
        update = get_length(document)
        fv.update(update)
        features.append(fv)
    return features

def get_length(document,tokenize=False):
    count = 0
    if tokenize:
        raise NotImplementedError

    if document.format == 'pdf':
        for page in document.pages:
            count += len(page)

    return {"character length": count}