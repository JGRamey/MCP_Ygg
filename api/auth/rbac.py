"""
Role-Based Access Control (RBAC) implementation for MCP Yggdrasil.
Provides fine-grained permission management with resource-level control.
"""
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging
from functools import lru_cache
import re

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources in the system."""
    CONCEPT = "concept"
    DOCUMENT = "document"
    CLAIM = "claim"
    RELATIONSHIP = "relationship"
    USER = "user"
    SYSTEM = "system"
    ANALYTICS = "analytics"
    SCRAPER = "scraper"
    DATABASE = "database"


class Action(Enum):
    """Possible actions on resources."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    APPROVE = "approve"
    REJECT = "reject"
    ANALYZE = "analyze"
    EXPORT = "export"
    MANAGE = "manage"


@dataclass
class Permission:
    """A permission defines an allowed action on a resource type."""
    resource: ResourceType
    action: Action
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.resource.value}.{self.action.value}"
    
    def matches(self, resource: ResourceType, action: Action, context: Dict[str, Any] = None) -> bool:
        """Check if this permission matches the requested resource and action."""
        if self.resource != resource or self.action != action:
            return False
        
        # Check conditions if any
        if self.conditions and context:
            for key, value in self.conditions.items():
                if key not in context or context[key] != value:
                    return False
        
        return True


@dataclass
class Role:
    """A role is a collection of permissions."""
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)  # Role inheritance
    
    def add_permission(self, permission: Permission):
        """Add a permission to this role."""
        self.permissions.add(permission)
    
    def has_permission(self, resource: ResourceType, action: Action, context: Dict[str, Any] = None) -> bool:
        """Check if this role has a specific permission."""
        return any(perm.matches(resource, action, context) for perm in self.permissions)


class RBACSystem:
    """
    Role-Based Access Control system with support for:
    - Role hierarchies
    - Resource-level permissions
    - Conditional permissions
    - Dynamic permission evaluation
    """
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}  # user_id -> set of role names
        self.resource_ownership: Dict[str, str] = {}  # resource_id -> owner_user_id
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize the default system roles."""
        # Admin role - full system access
        admin = Role(
            name="admin",
            description="System administrator with full access",
            permissions={
                Permission(resource_type, action)
                for resource_type in ResourceType
                for action in Action
            }
        )
        self.roles["admin"] = admin
        
        # Researcher role - create and analyze content
        researcher = Role(
            name="researcher",
            description="Can create and analyze content",
            permissions={
                # Read permissions
                Permission(ResourceType.CONCEPT, Action.READ),
                Permission(ResourceType.DOCUMENT, Action.READ),
                Permission(ResourceType.CLAIM, Action.READ),
                Permission(ResourceType.RELATIONSHIP, Action.READ),
                Permission(ResourceType.ANALYTICS, Action.READ),
                
                # Create permissions
                Permission(ResourceType.CONCEPT, Action.CREATE),
                Permission(ResourceType.DOCUMENT, Action.CREATE),
                Permission(ResourceType.CLAIM, Action.CREATE),
                Permission(ResourceType.RELATIONSHIP, Action.CREATE),
                
                # Analysis permissions
                Permission(ResourceType.CONCEPT, Action.ANALYZE),
                Permission(ResourceType.DOCUMENT, Action.ANALYZE),
                Permission(ResourceType.CLAIM, Action.ANALYZE),
                
                # Scraper permissions
                Permission(ResourceType.SCRAPER, Action.CREATE),
                Permission(ResourceType.SCRAPER, Action.READ),
            }
        )
        self.roles["researcher"] = researcher
        
        # Curator role - approve and manage content
        curator = Role(
            name="curator",
            description="Can approve, reject, and manage content",
            permissions={
                # All researcher permissions
                *researcher.permissions,
                
                # Additional curator permissions
                Permission(ResourceType.CONCEPT, Action.UPDATE),
                Permission(ResourceType.DOCUMENT, Action.UPDATE),
                Permission(ResourceType.CLAIM, Action.UPDATE),
                Permission(ResourceType.RELATIONSHIP, Action.UPDATE),
                
                Permission(ResourceType.CONCEPT, Action.APPROVE),
                Permission(ResourceType.DOCUMENT, Action.APPROVE),
                Permission(ResourceType.CLAIM, Action.APPROVE),
                
                Permission(ResourceType.CONCEPT, Action.REJECT),
                Permission(ResourceType.DOCUMENT, Action.REJECT),
                Permission(ResourceType.CLAIM, Action.REJECT),
                
                Permission(ResourceType.DATABASE, Action.EXPORT),
            }
        )
        curator.parent_roles.add("researcher")  # Inherits from researcher
        self.roles["curator"] = curator
        
        # Viewer role - read-only access
        viewer = Role(
            name="viewer",
            description="Read-only access to public content",
            permissions={
                Permission(ResourceType.CONCEPT, Action.READ),
                Permission(ResourceType.DOCUMENT, Action.READ),
                Permission(ResourceType.RELATIONSHIP, Action.READ),
                Permission(ResourceType.ANALYTICS, Action.READ),
            }
        )
        self.roles["viewer"] = viewer
        
        # System role - for automated processes
        system = Role(
            name="system",
            description="Automated system processes",
            permissions={
                Permission(resource_type, action)
                for resource_type in ResourceType
                for action in [Action.CREATE, Action.READ, Action.UPDATE, Action.ANALYZE]
            }
        )
        self.roles["system"] = system
        
        logger.info("Default roles initialized")
    
    def create_role(self, name: str, description: str, permissions: List[Permission] = None, 
                   parent_roles: List[str] = None) -> Role:
        """Create a new role."""
        if name in self.roles:
            raise ValueError(f"Role '{name}' already exists")
        
        role = Role(
            name=name,
            description=description,
            permissions=set(permissions or []),
            parent_roles=set(parent_roles or [])
        )
        
        self.roles[name] = role
        logger.info(f"Created role: {name}")
        return role
    
    def assign_role(self, user_id: str, role_name: str):
        """Assign a role to a user."""
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
        logger.info(f"Assigned role '{role_name}' to user {user_id}")
    
    def revoke_role(self, user_id: str, role_name: str):
        """Revoke a role from a user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
            logger.info(f"Revoked role '{role_name}' from user {user_id}")
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get all roles assigned to a user."""
        return self.user_roles.get(user_id, set())
    
    @lru_cache(maxsize=1024)
    def get_effective_permissions(self, user_id: str) -> Set[Permission]:
        """Get all effective permissions for a user (including inherited)."""
        permissions = set()
        
        # Get user's direct roles
        user_roles = self.get_user_roles(user_id)
        
        # Process each role and its parents
        processed_roles = set()
        roles_to_process = list(user_roles)
        
        while roles_to_process:
            role_name = roles_to_process.pop()
            if role_name in processed_roles:
                continue
            
            processed_roles.add(role_name)
            
            if role_name in self.roles:
                role = self.roles[role_name]
                permissions.update(role.permissions)
                
                # Add parent roles to process
                roles_to_process.extend(role.parent_roles)
        
        return permissions
    
    def check_permission(self, user_id: str, resource: ResourceType, action: Action, 
                        resource_id: Optional[str] = None, context: Dict[str, Any] = None) -> bool:
        """
        Check if a user has permission to perform an action on a resource.
        
        Args:
            user_id: User ID
            resource: Resource type
            action: Action to perform
            resource_id: Optional specific resource ID
            context: Optional context for conditional permissions
        
        Returns:
            True if permission is granted
        """
        # System bypass for automated processes
        if user_id == "system":
            return True
        
        # Check ownership if resource_id is provided
        if resource_id and resource_id in self.resource_ownership:
            owner_id = self.resource_ownership[resource_id]
            if owner_id == user_id and action in [Action.READ, Action.UPDATE, Action.DELETE]:
                return True
        
        # Get user's effective permissions
        permissions = self.get_effective_permissions(user_id)
        
        # Check for matching permission
        for perm in permissions:
            if perm.matches(resource, action, context):
                return True
        
        logger.warning(f"Permission denied: user={user_id}, resource={resource.value}, action={action.value}")
        return False
    
    def set_resource_owner(self, resource_id: str, owner_id: str):
        """Set the owner of a resource."""
        self.resource_ownership[resource_id] = owner_id
    
    def get_resource_owner(self, resource_id: str) -> Optional[str]:
        """Get the owner of a resource."""
        return self.resource_ownership.get(resource_id)
    
    def create_permission_string(self, resource: ResourceType, action: Action) -> str:
        """Create a permission string in format 'resource.action'."""
        return f"{resource.value}.{action.value}"
    
    def parse_permission_string(self, permission_str: str) -> Optional[Permission]:
        """Parse a permission string into a Permission object."""
        match = re.match(r'^(\w+)\.(\w+)$', permission_str)
        if not match:
            return None
        
        resource_str, action_str = match.groups()
        
        try:
            resource = ResourceType(resource_str)
            action = Action(action_str)
            return Permission(resource, action)
        except ValueError:
            return None
    
    def get_user_permissions_list(self, user_id: str) -> List[str]:
        """Get a list of permission strings for a user."""
        permissions = self.get_effective_permissions(user_id)
        return [str(perm) for perm in permissions]
    
    def has_any_permission(self, user_id: str, permission_strings: List[str]) -> bool:
        """Check if user has any of the specified permissions."""
        user_permissions = set(self.get_user_permissions_list(user_id))
        return bool(user_permissions.intersection(permission_strings))
    
    def has_all_permissions(self, user_id: str, permission_strings: List[str]) -> bool:
        """Check if user has all of the specified permissions."""
        user_permissions = set(self.get_user_permissions_list(user_id))
        return all(perm in user_permissions for perm in permission_strings)


# Global RBAC instance
rbac_system = RBACSystem()


# Decorators for FastAPI routes
from functools import wraps
from typing import Callable


def require_permission(resource: ResourceType, action: Action):
    """
    Decorator to require specific permission for an endpoint.
    
    Usage:
        @require_permission(ResourceType.DOCUMENT, Action.CREATE)
        async def create_document(user_id: str = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, user_id: str = None, **kwargs):
            if not user_id:
                raise ValueError("User ID is required")
            
            if not rbac_system.check_permission(user_id, resource, action):
                raise PermissionError(f"Permission denied: {resource.value}.{action.value}")
            
            return await func(*args, user_id=user_id, **kwargs)
        
        return wrapper
    return decorator


def require_any_role(roles: List[str]):
    """Decorator to require any of the specified roles."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, user_id: str = None, **kwargs):
            if not user_id:
                raise ValueError("User ID is required")
            
            user_roles = rbac_system.get_user_roles(user_id)
            if not user_roles.intersection(roles):
                raise PermissionError(f"Required roles: {', '.join(roles)}")
            
            return await func(*args, user_id=user_id, **kwargs)
        
        return wrapper
    return decorator