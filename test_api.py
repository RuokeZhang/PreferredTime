"""
API测试脚本
用于测试Movie Recommendation API的各个端点
"""

import requests
import json
import time

BASE_URL = "http://localhost:8082"


def print_separator(title=""):
    """打印分隔线"""
    print("\n" + "=" * 60)
    if title:
        print(f" {title}")
        print("=" * 60)
    print()


def test_root():
    """测试根路径"""
    print_separator("测试根路径")
    response = requests.get(f"{BASE_URL}/")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_health():
    """测试健康检查"""
    print_separator("测试健康检查")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.json().get('recommender_loaded', False)


def test_recommend(user_id):
    """测试推荐接口"""
    print_separator(f"测试推荐接口 - 用户 {user_id}")
    response = requests.get(f"{BASE_URL}/recommend/{user_id}")
    print(f"状态码: {response.status_code}")
    print(f"推荐的电影ID: {response.text}")
    
    # 解析并显示
    if response.status_code == 200:
        movie_ids = response.text.split(',')
        print(f"推荐数量: {len(movie_ids)}")
        print(f"前5个推荐: {', '.join(movie_ids[:5])}")


def test_user_profile(user_id):
    """测试用户画像"""
    print_separator(f"测试用户画像 - 用户 {user_id}")
    response = requests.get(f"{BASE_URL}/user/{user_id}/profile")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_similar_movies(movie_id, top_n=5):
    """测试相似电影"""
    print_separator(f"测试相似电影 - 电影 {movie_id}")
    response = requests.get(f"{BASE_URL}/movie/{movie_id}/similar?top_n={top_n}")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"电影ID: {data['movie_id']}")
        print(f"相似电影:")
        for item in data['similar_movies']:
            print(f"  - 电影 {item['movie_id']}: 相似度 {item['similarity']:.4f}")


def test_reload():
    """测试重新加载模型"""
    print_separator("测试重新加载模型")
    response = requests.post(f"{BASE_URL}/reload")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 60)
    print(" Movie Recommendation API 综合测试")
    print("=" * 60)
    
    try:
        # 1. 测试根路径
        test_root()
        time.sleep(1)
        
        # 2. 测试健康检查
        recommender_loaded = test_health()
        time.sleep(1)
        
        if not recommender_loaded:
            print("\n⚠️  推荐系统未加载，可能需要生成测试数据")
            print("请运行: python3 test_producer.py")
            print("然后选择选项3生成真实数据集")
            return
        
        # 3. 测试多个用户的推荐
        test_user_ids = [1, 5, 10, 50, 99]
        for user_id in test_user_ids:
            test_recommend(user_id)
            time.sleep(0.5)
        
        # 4. 测试用户画像
        test_user_profile(1)
        time.sleep(1)
        
        # 5. 测试相似电影
        test_similar_movies(1, top_n=5)
        time.sleep(1)
        
        test_similar_movies(10, top_n=5)
        time.sleep(1)
        
        # 6. 测试重新加载模型
        test_reload()
        time.sleep(2)
        
        # 7. 验证重新加载后的推荐
        print_separator("验证重新加载后的推荐")
        test_recommend(1)
        
        print_separator("测试完成")
        print("✅ 所有测试已完成！")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 连接失败！")
        print("请确保API服务正在运行:")
        print("  ./run.sh")
        print("  或")
        print("  ./start_api.sh")
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")


def interactive_test():
    """交互式测试"""
    print("=" * 60)
    print(" Movie Recommendation API 交互式测试")
    print("=" * 60)
    
    while True:
        print("\n选择测试:")
        print("1. 测试根路径")
        print("2. 健康检查")
        print("3. 获取推荐")
        print("4. 用户画像")
        print("5. 相似电影")
        print("6. 重新加载模型")
        print("7. 运行综合测试")
        print("8. 退出")
        
        choice = input("\n请选择 (1-8): ").strip()
        
        try:
            if choice == '1':
                test_root()
            elif choice == '2':
                test_health()
            elif choice == '3':
                user_id = int(input("请输入用户ID: "))
                test_recommend(user_id)
            elif choice == '4':
                user_id = int(input("请输入用户ID: "))
                test_user_profile(user_id)
            elif choice == '5':
                movie_id = int(input("请输入电影ID: "))
                top_n = int(input("请输入要返回的相似电影数量 (默认5): ") or "5")
                test_similar_movies(movie_id, top_n)
            elif choice == '6':
                test_reload()
            elif choice == '7':
                run_comprehensive_test()
            elif choice == '8':
                print("退出...")
                break
            else:
                print("无效的选择，请重试")
        
        except requests.exceptions.ConnectionError:
            print("\n❌ 连接失败！请确保API服务正在运行")
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
        except Exception as e:
            print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'auto':
        # 自动运行综合测试
        run_comprehensive_test()
    else:
        # 交互式测试
        interactive_test()


