from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker
from pymilvus import WeightedRanker
from pymilvus import MilvusClient, AsyncMilvusClient
import traceback
import asyncio
from FlagEmbedding import BGEM3FlagModel

class BGEM3Embedding:
    def __init__(self, model_path='/home/25-fengyuan/LLModel/bge-m3', use_fp16=True, device='cuda:6'):
        print(f"正在加载 BGE-M3 模型: {model_path}")
        print(f"使用设备: {device}")
        self.model = BGEM3FlagModel(
            model_path, use_fp16=use_fp16, device=device)
        self.pool = None
        print("✓ BGE-M3 模型加载完成")
    def encode(self, texts):
        """编码文档（用于存储到向量数据库）"""
        if isinstance(texts, str):
            texts = [texts]

        # 使用 encode_single_device 避免多进程
        embeddings = self.model.encode_single_device(
            sentences=texts,
            batch_size=12,
            max_length=8192,
            return_dense = True,
            return_sparse = True,
        )
        return embeddings
class hybridSearch():
    def __init__(self, cache_db = 'Query_Cache',
                 search_db = 'Law_Knowledge',
                 cache_collection = 'cache1',
                 search_collection = 'pdf_chunck_with_metadata',
                 embedding_model:BGEM3Embedding = None
                 ):
        self.cache_db = cache_db
        self.cache_collection = cache_collection
        self.search_db = search_db
        self.search_collection = search_collection
        # self.cacheClient = AsyncMilvusClient(uri='http://localhost:19530', db_name = self.cache_db)
        # self.searchClient = AsyncMilvusClient(uri='http://localhost:19530', db_name = self.search_db)
        self.cacheClient = None
        self.searchClient = None
        self.embedding_model = embedding_model
        # print(f"正在加载集合到内存...")
        # self.cacheClient.load_collection(collection_name=self.cache_collection)
        # self.searchClient.load_collection(collection_name=self.search_collection)
        # print(f"✓ 已加载集合: {self.cache_collection}, {self.search_collection}")
    async def _ensure_clients(self):
        """确保异步客户端在事件循环中正确初始化"""
        if self.cacheClient is None:
            self.cacheClient = AsyncMilvusClient(uri='http://localhost:19530', db_name=self.cache_db)
            print(f"✓ Cache 客户端已连接")
        if self.searchClient is None:
            self.searchClient = AsyncMilvusClient(uri='http://localhost:19530', db_name=self.search_db)
            print(f"✓ Search 客户端已连接")
    async def get_all_vec(self, ques: str)->dict: # Cache 机制,
        await self._ensure_clients()

        # 先尝试cache
        ans = await self.cacheClient.query(
            collection_name=self.cache_collection,
            filter=f'question in ["{ques}"]',
            output_fields=['id', 'dense_vector', 'sparse_vector']
        )
        if len(ans) > 0:
            return ans[0]
        # 否则进行向量检索，并存储在缓冲库中
        try:
            print('缓存未命中，正在嵌入')
            ans = await asyncio.to_thread(self.embedding_model.encode, ques)
            # print(ans)
            # 因为cache没有，所以我存储一个
            text_dense = ans['dense_vecs']
            text_sparse = ans['lexical_weights']

            text_dense_list = text_dense.tolist()[0]
            text_sparse_dict = dict(text_sparse[0])
            await self.cacheClient.insert(data = [{'question': ques, 'dense_vector':text_dense_list, 'sparse_vector':text_sparse_dict}], collection_name = self.cache_collection)
            return {'dense_vector': text_dense_list, 'sparse_vector': text_sparse_dict}
        except Exception as e:
            print(f'报错:{e}')
            traceback.print_exc()
            raise ConnectionError(f'向量嵌入模型链接失败,请查看网络') from e
    async def __call__(self, ques: str="请介绍一下'海贝思'引起的泥石流相关事件", topk = 20, spare_topk = 50, dense_topk = 50):
        self.topk = topk
        # 混合检索
        ans = await self.get_all_vec(ques)
        query_dense_vector = ans['dense_vector']
        search_param_1 = {
            "data": [query_dense_vector],
            "anns_field": "text_dense",
            "param": {
                "params": {
                    "M": 64,
                    "efConstruction": 500
                }
            },  # Index building params
            "limit": dense_topk
        }
        request_1 = AnnSearchRequest(**search_param_1)

        query_sparse_vector = ans['sparse_vector']
        search_param_2 = {
            "data": [query_sparse_vector],
            "anns_field": "text_sparse",
            "param": {
                "metric_type": "IP",
                "params": {}
            },
            "limit": spare_topk
        }
        request_2 = AnnSearchRequest(**search_param_2)
        reqs = [request_1, request_2]

        # ranker = WeightedRanker(0.8, 0.5)
        ranker = RRFRanker(60)

        milvus_client = self.searchClient
        res = await milvus_client.hybrid_search(
            collection_name=self.search_collection,
            reqs=reqs,
            ranker=ranker,
            limit=self.topk
        )

        contents = []
        for hits in res:
            print("TopK results:")
            for hit in hits:
                print(hit)
                content = await milvus_client.get(
                    collection_name=self.search_collection,
                    ids=hit['id'],
                    output_fields=["raw_data", "title", 'page', 'total_page']
                )
                contents += [content[0]]# [{}]
                # print(type(content))
                # print([content[0]])
                print([content[0]['raw_data'], content[0]['title'], content[0]['page'], content[0]['total_page']])
        return contents

# if __name__ == '__main__':
#     bgem3_embedding = BGEM3Embedding()
#     mysearch = hybridSearch(embedding_model=bgem3_embedding)
#     asyncio.run(mysearch(ques = '这个江苏省共有滑坡、 崩塌隐患点多少处？！'))

    