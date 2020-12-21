def augment_inner_product_matrix(factors):
    """
    This involves transforming each row by adding one extra dimension as suggested in the paper:
    "Speeding Up the Xbox Recommender System Using a Euclidean Transformation for Inner-Product
    Spaces" https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf
    
    # Code adopted from 'implicit' repo
    # https://github.com/benfred/implicit/blob/4dba6dd90c4a470cb25ede34a930c56558ef10b2/implicit/approximate_als.py#L37
    """
    import numpy as np
    
    norms = np.linalg.norm(factors, axis=1)
    max_norm = norms.max()

    extra_dimension = np.sqrt(max_norm ** 2 - norms ** 2)
    return np.append(factors, extra_dimension.reshape(norms.shape[0], 1), axis=1)

def nmslib_recommend(spark, df, model, k=500):
    
    import nmslib
    import numpy as np
    
    # user_factors only for the users in the given df (ordered by user id)
    subset_user = df.select('user').distinct()
    whole_user = model.userFactors
    user = whole_user.join(subset_user, whole_user.id == subset_user.user).orderBy('id')
    user_factors = user.select('features')
    
    # item_factors ordered by item id
    item = model.itemFactors.orderBy('id')
    item_factors = item.select('features')
    
    # save user/item label
    user_label = [user[0] for user in user.select('id').collect()]
    item_label = [item[0] for item in item.select('id').collect()]
    
    # to numpy array
    user_factors = np.array(user_factors.collect()).reshape(-1, model.rank)
    item_factors = np.array(item_factors.collect()).reshape(-1, model.rank)
    print("feature array created")

    # Euclidean Transformation for Inner-Product Spaces
    extra = augment_inner_product_matrix(item_factors)
    print("augmented")
    
    # create nmslib index
    recommend_index = nmslib.init(method='hnsw', space='cosinesimil')
    recommend_index.addDataPointBatch(extra)
    recommend_index.createIndex({'post': 2})
    print("index created")
    
    # recommend for given users
    query = np.append(user_factors, np.zeros((user_factors.shape[0],1)), axis=1)
    results = recommend_index.knnQueryBatch(query, k)
    
    recommend = []
    for user_ in range(len(results)):
        itemlist = []
        for item_ in results[user_][0]:
            itemlist.append(item_label[item_])
        recommend.append((user_label[user_], itemlist))
        
    return recommend
