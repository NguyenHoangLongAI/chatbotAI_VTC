#!/usr/bin/env python3
"""
Streaming Chat Client - Test streaming API
Usage: python streaming_client.py
"""

import requests
import json
import sys
from typing import List, Dict
import time


class StreamingChatClient:
    def __init__(self, base_url: str = "http://localhost:8501/"):
        self.base_url = base_url
        self.session = requests.Session()
        self.chat_history = []

    def check_health(self):
        """Kiểm tra tình trạng API"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"🟢 API Status: {health_data['status']}")
                print(f"📊 Message: {health_data['message']}")
                return True
            else:
                print(f"🔴 API Error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return False

    def send_message_streaming(self, question: str) -> None:
        try:
            payload = {
                "question": question,
                "history": self.chat_history,
                "stream": True
            }

            print(f"\n❓ Câu hỏi: {question}")
            print("💬 Trả lời: ", end='', flush=True)

            start_time = time.time()
            first_chunk_time = None

            full_answer = ""
            references = []

            with self.session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    stream=True,
                    timeout=60
            ) as response:

                if response.status_code != 200:
                    print(f"\n🔴 Error: {response.status_code}")
                    return

                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode("utf-8")

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]

                    try:
                        chunk_data = json.loads(data_str)
                        chunk_type = chunk_data.get("type")

                        if chunk_type == "chunk":
                            # ⏱️ FIRST CHUNK TIMING
                            if first_chunk_time is None:
                                first_chunk_time = time.time()
                                latency = first_chunk_time - start_time - 1

                            content = chunk_data.get("content", "")
                            print(content, end="", flush=True)
                            full_answer += content

                        elif chunk_type == "references":
                            references = chunk_data.get("references", [])

                        elif chunk_type == "end":
                            print("\n\n📊 Status:", chunk_data.get("status", "SUCCESS"))

                    except json.JSONDecodeError:
                        continue

            end_time = time.time()

            # Update history
            self.chat_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer}
            ])

            print(f"⏱️  Total streaming time: {latency:.2f}s")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ Error: {e}")

    def send_message_non_streaming(self, question: str) -> None:
        """
        Gửi câu hỏi với non-streaming mode (original)
        """
        try:
            payload = {
                "question": question,
                "history": self.chat_history,
                "stream": False  # Disable streaming
            }

            print(f"\n❓ Câu hỏi: {question}")
            print("⏳ Đang xử lý...")

            start_time = time.time()

            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=60
            )

            end_time = time.time()

            if response.status_code == 200:
                result = response.json()

                print(f"\n💬 Trả lời:\n{result['answer']}")
                print(f"\n⏱️  Thời gian: {end_time - start_time:.2f}s")
                print(f"📊 Status: {result.get('status', 'UNKNOWN')}")

                # Update history
                self.chat_history.append({"role": "user", "content": question})
                self.chat_history.append({"role": "assistant", "content": result['answer']})

                # Display references
                if result.get("references"):
                    print(f"\n📚 Tài liệu tham khảo:")
                    for i, ref in enumerate(result["references"], 1):
                        print(f"  {i}. {ref['type']}: {ref['document_id']}")

                print("=" * 60)
            else:
                print(f"🔴 Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def compare_streaming_vs_non_streaming(self, question: str):
        """So sánh streaming vs non-streaming"""
        print("\n" + "=" * 60)
        print("🔬 COMPARISON: STREAMING vs NON-STREAMING")
        print("=" * 60)

        # Test 1: Non-streaming
        print("\n[1] NON-STREAMING MODE:")
        print("-" * 60)
        self.send_message_non_streaming(question)

        # Clear history for fair comparison
        self.chat_history.clear()

        # Test 2: Streaming
        print("\n[2] STREAMING MODE:")
        print("-" * 60)
        self.send_message_streaming(question)

        print("\n" + "=" * 60)
        print("✅ COMPARISON COMPLETE")
        print("=" * 60)

    def interactive_mode(self):
        """Chế độ chat tương tác"""
        print("🚀 Streaming Chat Client Started!")
        print("-" * 50)

        if not self.check_health():
            print("❌ Không thể kết nối tới API!")
            return

        print("\n💡 Commands:")
        print("  /stream   - Gửi câu hỏi với streaming")
        print("  /normal   - Gửi câu hỏi không streaming")
        print("  /compare  - So sánh streaming vs non-streaming")
        print("  /history  - Xem lịch sử")
        print("  /clear    - Xóa lịch sử")
        print("  /quit     - Thoát")
        print("\n" + "=" * 50)

        streaming_mode = True  # Default: streaming

        while True:
            try:
                mode_indicator = "🔄 STREAMING" if streaming_mode else "📋 NORMAL"
                question = input(f"\n[{mode_indicator}] ❓ Câu hỏi: ").strip()

                if not question:
                    continue

                # Commands
                if question == "/quit":
                    print("👋 Tạm biệt!")
                    break
                elif question == "/stream":
                    streaming_mode = True
                    print("✅ Switched to STREAMING mode")
                elif question == "/normal":
                    streaming_mode = False
                    print("✅ Switched to NON-STREAMING mode")
                elif question == "/compare":
                    test_q = input("Câu hỏi để test: ").strip()
                    if test_q:
                        self.compare_streaming_vs_non_streaming(test_q)
                elif question == "/history":
                    if self.chat_history:
                        print("\n📜 Lịch sử chat:")
                        for msg in self.chat_history:
                            role = "👤" if msg["role"] == "user" else "🤖"
                            print(f"{role} {msg['content'][:100]}...")
                    else:
                        print("📝 Chưa có lịch sử")
                elif question == "/clear":
                    self.chat_history.clear()
                    print("🗑️  Đã xóa lịch sử")
                else:
                    # Send question
                    if streaming_mode:
                        self.send_message_streaming(question)
                    else:
                        self.send_message_non_streaming(question)

            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Single question mode
        client = StreamingChatClient()
        question = " ".join(sys.argv[1:])

        if client.check_health():
            print("\n🔬 Testing both modes:\n")
            client.compare_streaming_vs_non_streaming(question)
    else:
        # Interactive mode
        client = StreamingChatClient()
        client.interactive_mode()


if __name__ == "__main__":
    main()