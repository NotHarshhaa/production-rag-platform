"""Server-side conversation sessions: POST/GET/PATCH/DELETE /conversations.

Conversations store chat turns per tenant so clients can continue a thread
by sending ``conversation_id`` on /query (or /query/stream) instead of
replaying the full message history themselves.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import Tenant
from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import (
    CitationOut,
    ConversationCreateRequest,
    ConversationDetailOut,
    ConversationMessageOut,
    ConversationOut,
    ConversationRenameRequest,
)
from app.storage.repository import ConversationRepository

router = APIRouter(prefix="/conversations", tags=["conversations"])

_Container = Annotated[ServiceContainer, Depends(get_container)]


def _not_found(conversation_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Conversation {conversation_id} not found",
    )


def _message_out(message) -> ConversationMessageOut:
    return ConversationMessageOut(
        id=message.id,
        role=message.role,
        content=message.content,
        citations=[
            CitationOut.model_validate(citation)
            for citation in ConversationRepository.citations_of(message)
        ],
        created_at=message.created_at,
    )


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreateRequest | None, container: _Container, tenant: Tenant
) -> ConversationOut:
    """Start an empty conversation session."""

    conversation = container.conversations.create(
        title=request.title if request else None, tenant_id=tenant
    )
    return ConversationOut(
        id=conversation.id,
        title=conversation.title,
        tenant_id=conversation.tenant_id,
        message_count=0,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("", response_model=list[ConversationOut])
def list_conversations(container: _Container, tenant: Tenant) -> list[ConversationOut]:
    """List the tenant's conversations, most recently active first."""

    return [
        ConversationOut(
            id=conversation.id,
            title=conversation.title,
            tenant_id=conversation.tenant_id,
            message_count=container.conversations.message_count(conversation.id),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
        for conversation in container.conversations.list(tenant)
    ]


@router.get("/{conversation_id}", response_model=ConversationDetailOut)
def get_conversation(
    conversation_id: str, container: _Container, tenant: Tenant
) -> ConversationDetailOut:
    """Fetch one conversation with all of its messages."""

    conversation = container.conversations.get(conversation_id, tenant)
    if conversation is None:
        raise _not_found(conversation_id)
    messages = container.conversations.list_messages(conversation_id)
    return ConversationDetailOut(
        id=conversation.id,
        title=conversation.title,
        tenant_id=conversation.tenant_id,
        message_count=len(messages),
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[_message_out(message) for message in messages],
    )


@router.patch("/{conversation_id}", response_model=ConversationOut)
def rename_conversation(
    conversation_id: str,
    request: ConversationRenameRequest,
    container: _Container,
    tenant: Tenant,
) -> ConversationOut:
    """Rename a conversation."""

    conversation = container.conversations.get(conversation_id, tenant)
    if conversation is None:
        raise _not_found(conversation_id)
    renamed = container.conversations.rename(conversation_id, request.title)
    return ConversationOut(
        id=renamed.id,
        title=renamed.title,
        tenant_id=renamed.tenant_id,
        message_count=container.conversations.message_count(conversation_id),
        created_at=renamed.created_at,
        updated_at=renamed.updated_at,
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str, container: _Container, tenant: Tenant
) -> None:
    """Delete a conversation and its messages."""

    conversation = container.conversations.get(conversation_id, tenant)
    if conversation is None:
        raise _not_found(conversation_id)
    container.conversations.delete(conversation_id)
