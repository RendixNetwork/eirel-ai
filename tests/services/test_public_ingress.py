from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from orchestration.consumer_api.main import app as consumer_api_app
from shared.common.models import ConsumerSessionState, TaskRequestRecord


async def test_consumer_api_reads_task_and_session(monkeypatch):
    monkeypatch.setenv("CONSUMER_API_KEYS", "consumer-secret")

    async with consumer_api_app.router.lifespan_context(consumer_api_app):
        db = consumer_api_app.state.execution_store.db
        with db.sessionmaker() as session:
            session.add(
                ConsumerSessionState(
                    session_id="session-1",
                    user_id="user-1",
                    latest_task_id="task-1",
                    messages_json=[{"role": "user", "content": "hello"}],
                )
            )
            session.add(
                TaskRequestRecord(
                    task_id="task-1",
                    session_id="session-1",
                    user_id="user-1",
                    raw_input="hello",
                    mode="sync",
                    status="queued",
                    queue_state="queued",
                    constraints_json={},
                    metadata_json={},
                )
            )
            session.commit()
        transport = ASGITransport(app=consumer_api_app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            task_response = await client.get(
                "/v1/tasks/task-1",
                headers={"X-API-Key": "consumer-secret"},
            )
            session_response = await client.get(
                "/v1/sessions/session-1",
                headers={"X-API-Key": "consumer-secret"},
            )
            assert task_response.status_code == 200
            assert session_response.status_code == 200
            assert task_response.json()["task_id"] == "task-1"
            assert session_response.json()["session_id"] == "session-1"
