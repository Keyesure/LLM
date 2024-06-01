import json
from queue import Queue
from threading import Thread
from flask import Flask, request
from flask_cors import CORS
from my_openai import run_react_agent

app = Flask(__name__)
CORS(app, resources=r'/*')
send_q = Queue(maxsize=0)
recv_q = Queue(maxsize=0)
error_q = Queue(maxsize=0)


@app.before_first_request
def work():
    def run():
        chat_history = []
        while True:
            try:
                message = send_q.get(block=True, timeout=None)  # block
                output, chat_history = run_react_agent(message, chat_history)
                error_q.put(False)
                recv_q.put(output)
            except Exception as e:
                error_q.put(True)
                recv_q.put(str(e))
            finally:
                continue

    thread = Thread(target=run)
    thread.start()


@app.route('/message', methods=['POST'])
def get_message():
    message = request.get_data()
    item = json.loads(message, strict=False)
    print(item['message'])
    send_q.put(item['message'])
    feedback = recv_q.get(timeout=None)
    error = error_q.get(timeout=None)
    if error:
        return {"reply": feedback, "status": "发送失败"}
    return {'reply': feedback, 'status': '发送成功'}


if __name__ == '__main__':
    app.run(port=8081)
