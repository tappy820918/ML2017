
#MF model
def get_model(n_users, n_items, latent_dim = 6666):
    user_input = Input(shape = [1])
    item_input = Input(shape = [1])
    user_vec = Embedding(n_users,latent_dim,embeddings_initializer = 'random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items,latent_dim,embeddings_initializer = 'random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users,1,embeddings_initializer = 'zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items,1,embeddings_initializer = 'zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes = 1)([user_vec,item_vec])
    r_hat = Add()([r_hat,user_bias,item_bias])
    model = keras.models.Model([user_input,item_input],r_hat)
    model.compile(loss = 'mse',optimizer = 'sgd')
    return(model)

#NN model
def nn_model(n_users, n_items, latent_dim = 6666):
    user_input = Input(shape = [1])
    item_input = Input(shape = [1])
    user_vec = Embedding(n_users,latent_dim,embeddings_initializer = 'random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items,latent_dim,embeddings_initializer = 'random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    merge_vec = Concatenate()([user_vec,item_vec])
    hidden = Dense(150,activation = 'relu')(merge_vec)
    hidden = Dense(50,activation = 'relu')(hidden)
    output = Dense(1)(hidden)
    model = keras.models.Model([user_input,item_input],output)
    model.compile(loss = 'mse',optimizer = 'sgd')
    model.summary()
    return(model)

#get embedding
user_emb = np.array(model.layers[2].get_weights()).squeeze()
print('user embedding shape:', user_emb.shape)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print('user embedding shape:', movie_emb.shape)
np.save('user_emb.npy',user_emb)
np.save('movie_emb.npy',movie_emb)

def draw(x,y):
	from matplotlib import pyplot as plt
	from tsne import bh_sne
	y = np.array(y)
	x = np.array(x,dtype=np.flaot64)
	#perform t-SNE embedding
	vis_data = bh_sne(x)
	#plot the result
	vis_x = vis_data[:,0]
	vis_y = vis_data[:,1]

	cm = plt.cm.get_cmap('RdYLBu')
	sc = plt.scatter(vis_x,vis_y,c=y,cmap=cm)
	plt.colorbar(sc)
	plt.show()

#(sklearn version of tsne)
def sk_draw(x,y):
	from matplotlib import pyplot as plt
	from sklearn.manifold import TSNE
	y = np.array(y)
	x = np.array(x,dtype=np.float64)
	#perform t-SNE embedding
	vis_data = TSNE(n_components=2).fit_transform(x)
	#plot the result
	vis_x = vis_data[:,0]
	vis_y = vis_data[:,1]
	cm = plt.cm.get_cmap('RdYLBu')
	sc = plt.scatter(vis_x,vis_y,c=y,cmap=cm)
	plt.colorbar(sc)
	plt.show()
