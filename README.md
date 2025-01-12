# Học CrewAI - Xây dựng AI Assistant

Đây là dự án học tập về CrewAI - một framework mới để xây dựng AI Assistant. Dự án gồm 2 ứng dụng để thực hành các khái niệm khác nhau của CrewAI.

## 🎯 Mục tiêu học tập

- Tìm hiểu về CrewAI và cách xây dựng AI Assistant
- Thực hành với các khái niệm: Agents, Tasks, Crew
- Tích hợp CrewAI với Streamlit để tạo giao diện người dùng
- Làm việc với OpenAI GPT và LangChain

## 📚 Cấu trúc dự án

### 1. Agents_and_Tasks/
Folder này chứa ứng dụng đầu tiên để học về cách tạo và kết hợp các Agents:

- `crewai_agents_callback.py`: 
  - Tạo 2 agents: AIExpert và AIBlogger
  - Thực hiện 3 tasks: tạo keywords, tạo outline và viết blog
  - Học cách sử dụng callback function để theo dõi kết quả

- `output.txt`: File lưu kết quả output của các agents

### 2. Conservation_Film_Bot/
Folder này chứa ứng dụng chatbot phức tạp hơn:

- `app.py`: 
  - Giao diện chatbot sử dụng Streamlit
  - Học cách tạo chat interface
  - Quản lý lịch sử chat

- `crew.py`:
  - Cấu hình CrewAI
  - Học cách tạo crew từ file config

- `config/`:
  - `agents.yaml`: Cấu hình chi tiết cho agent
  - `tasks.yaml`: Cấu hình nhiệm vụ của agent