import fnmatch
import json
import logging
import os
import shutil
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class PresetRepository:
    """
    Repository for managing presets using a file-based storage approach.
    
    This class handles the storage and retrieval of presets as JSON files,
    with support for user-specific presets and public/shared presets.
    It implements caching for improved performance.
    """
    
    # Default directories for storing presets
    DEFAULT_USER_PRESET_DIR = "presets/user"
    DEFAULT_PUBLIC_PRESET_DIR = "presets/public"
    PRESET_FILE_EXTENSION = ".json"
    
    def __init__(self, base_dir: str = None, 
                 user_preset_dir: str = None, 
                 public_preset_dir: str = None,
                 cache_size: int = 100):
        """
        Initialize the PresetRepository.
        
        Args:
            base_dir (str, optional): Base directory for storing presets.
                                     Defaults to current working directory.
            user_preset_dir (str, optional): Subdirectory for user presets.
                                            Defaults to DEFAULT_USER_PRESET_DIR.
            public_preset_dir (str, optional): Subdirectory for public presets.
                                             Defaults to DEFAULT_PUBLIC_PRESET_DIR.
            cache_size (int, optional): Size of the LRU cache. Defaults to 100.
        """
        self.base_dir = base_dir or os.getcwd()
        self.user_preset_dir = user_preset_dir or self.DEFAULT_USER_PRESET_DIR
        self.public_preset_dir = public_preset_dir or self.DEFAULT_PUBLIC_PRESET_DIR
        self.cache_size = cache_size
        
        # Ensure preset directories exist
        self._ensure_directories()
        
        # Thread lock for concurrent access
        self._lock = threading.RLock()
        
        # In-memory cache of loaded presets
        self._preset_cache = {}
        
        # Flag to track if the preset list is dirty (needs refreshing)
        self._preset_list_dirty = True
        
        # Cached list of preset metadata
        self._preset_list_cache = []
        
        logger.info(f"PresetRepository initialized with base directory: {self.base_dir}")
    
    def _ensure_directories(self) -> None:
        """Ensure that the preset directories exist."""
        os.makedirs(os.path.join(self.base_dir, self.user_preset_dir), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, self.public_preset_dir), exist_ok=True)
        logger.debug(f"Preset directories created (if they didn't exist)")
    
    def _get_preset_path(self, preset_id: str, user_id: str = None) -> str:
        """
        Get the full path for a preset file.
        
        Args:
            preset_id (str): ID of the preset.
            user_id (str, optional): User ID for user-specific presets. 
                                    If None, the preset is considered public.
        
        Returns:
            str: Full path to the preset file.
        """
        if user_id:
            user_dir = os.path.join(self.base_dir, self.user_preset_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)
            return os.path.join(user_dir, f"{preset_id}{self.PRESET_FILE_EXTENSION}")
        else:
            return os.path.join(self.base_dir, self.public_preset_dir, 
                               f"{preset_id}{self.PRESET_FILE_EXTENSION}")
    
    def save_preset(self, preset_data: Dict[str, Any], 
                   user_id: str = None, overwrite: bool = True) -> str:
        """
        Save a preset to storage.
        
        Args:
            preset_data (Dict[str, Any]): The preset data to save.
            user_id (str, optional): User ID for user-specific presets.
                                    If None, the preset is saved as public.
            overwrite (bool, optional): Whether to overwrite if preset_id already exists.
                                      Defaults to True.
        
        Returns:
            str: ID of the saved preset.
        
        Raises:
            ValueError: If the preset data is invalid or if a preset with the same ID 
                       exists and overwrite is False.
        """
        if not preset_data or not isinstance(preset_data, dict):
            raise ValueError("Invalid preset data. Expected a dictionary.")
        
        preset_id = preset_data.get("id")
        if not preset_id:
            raise ValueError("Preset data must contain an 'id' field.")
        
        with self._lock:
            preset_path = self._get_preset_path(preset_id, user_id)
            
            # Check if preset already exists and overwrite flag is False
            if not overwrite and os.path.exists(preset_path):
                raise ValueError(f"Preset with ID '{preset_id}' already exists.")
            
            # Add metadata if not present
            if "metadata" not in preset_data:
                preset_data["metadata"] = {}
            
            preset_data["metadata"]["last_modified"] = datetime.now().isoformat()
            preset_data["metadata"]["user_id"] = user_id
            
            # Save to file
            try:
                with open(preset_path, 'w', encoding='utf-8') as f:
                    json.dump(preset_data, f, indent=2)
                
                # Update cache
                cache_key = self._get_cache_key(preset_id, user_id)
                self._preset_cache[cache_key] = preset_data
                self._preset_list_dirty = True
                
                logger.info(f"Preset '{preset_id}' saved successfully.")
                return preset_id
            
            except Exception as e:
                logger.error(f"Error saving preset '{preset_id}': {str(e)}")
                raise
    
    def load_preset(self, preset_id: str, user_id: str = None) -> Dict[str, Any]:
        """
        Load a preset from storage.
        
        Args:
            preset_id (str): ID of the preset to load.
            user_id (str, optional): User ID for user-specific presets.
                                    If None, loads from public presets.
        
        Returns:
            Dict[str, Any]: The loaded preset data.
        
        Raises:
            FileNotFoundError: If the preset doesn't exist.
            ValueError: If the preset data is invalid.
        """
        cache_key = self._get_cache_key(preset_id, user_id)
        
        # Try to get from cache first
        if cache_key in self._preset_cache:
            logger.debug(f"Preset '{preset_id}' loaded from cache.")
            return self._preset_cache[cache_key]
        
        # Not in cache, load from file
        with self._lock:
            preset_path = self._get_preset_path(preset_id, user_id)
            
            if not os.path.exists(preset_path):
                # If not found in user presets, try public presets
                if user_id:
                    try:
                        return self.load_preset(preset_id, None)
                    except FileNotFoundError:
                        pass
                
                logger.error(f"Preset '{preset_id}' not found.")
                raise FileNotFoundError(f"Preset with ID '{preset_id}' not found.")
            
            try:
                with open(preset_path, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                
                # Update cache
                self._preset_cache[cache_key] = preset_data
                
                # Manage cache size
                if len(self._preset_cache) > self.cache_size:
                    # Remove oldest items (simple approach)
                    keys_to_remove = list(self._preset_cache.keys())[:-self.cache_size]
                    for key in keys_to_remove:
                        del self._preset_cache[key]
                
                logger.debug(f"Preset '{preset_id}' loaded from file.")
                return preset_data
            
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing preset '{preset_id}': {str(e)}")
                raise ValueError(f"Invalid JSON in preset file: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error loading preset '{preset_id}': {str(e)}")
                raise
    
    def delete_preset(self, preset_id: str, user_id: str = None) -> bool:
        """
        Delete a preset from storage.
        
        Args:
            preset_id (str): ID of the preset to delete.
            user_id (str, optional): User ID for user-specific presets.
                                    If None, deletes from public presets.
        
        Returns:
            bool: True if preset was deleted, False if it didn't exist.
        
        Raises:
            PermissionError: If the preset exists but could not be deleted.
        """
        with self._lock:
            preset_path = self._get_preset_path(preset_id, user_id)
            
            if not os.path.exists(preset_path):
                logger.warning(f"Preset '{preset_id}' not found for deletion.")
                return False
            
            try:
                os.remove(preset_path)
                
                # Remove from cache
                cache_key = self._get_cache_key(preset_id, user_id)
                if cache_key in self._preset_cache:
                    del self._preset_cache[cache_key]
                
                self._preset_list_dirty = True
                logger.info(f"Preset '{preset_id}' deleted successfully.")
                return True
            
            except Exception as e:
                logger.error(f"Error deleting preset '{preset_id}': {str(e)}")
                raise
    
    def list_presets(self, user_id: str = None, include_public: bool = True, 
                    include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        List available presets.
        
        Args:
            user_id (str, optional): Filter presets for a specific user.
                                    If None, only public presets are returned.
            include_public (bool, optional): Whether to include public presets.
                                           Defaults to True.
            include_metadata (bool, optional): Whether to include metadata in results.
                                             Defaults to True.
        
        Returns:
            List[Dict[str, Any]]: List of preset data or metadata.
        """
        # Only regenerate list if something has changed or it's not cached
        if self._preset_list_dirty or not self._preset_list_cache:
            with self._lock:
                presets = []
                
                # Get public presets if requested
                if include_public:
                    public_preset_dir = os.path.join(self.base_dir, self.public_preset_dir)
                    if os.path.exists(public_preset_dir):
                        for file_name in os.listdir(public_preset_dir):
                            if file_name.endswith(self.PRESET_FILE_EXTENSION):
                                preset_id = file_name[:-len(self.PRESET_FILE_EXTENSION)]
                                try:
                                    preset_data = self.load_preset(preset_id, None)
                                    if include_metadata:
                                        presets.append(preset_data)
                                    else:
                                        # Only include basic info without full configuration
                                        presets.append({
                                            "id": preset_id,
                                            "name": preset_data.get("name", "Unnamed Preset"),
                                            "tags": preset_data.get("tags", []),
                                            "metadata": preset_data.get("metadata", {})
                                        })
                                except Exception as e:
                                    logger.warning(f"Error loading preset '{preset_id}': {str(e)}")
                
                # Get user presets if a user_id is provided
                if user_id:
                    user_preset_dir = os.path.join(self.base_dir, self.user_preset_dir, user_id)
                    if os.path.exists(user_preset_dir):
                        for file_name in os.listdir(user_preset_dir):
                            if file_name.endswith(self.PRESET_FILE_EXTENSION):
                                preset_id = file_name[:-len(self.PRESET_FILE_EXTENSION)]
                                try:
                                    preset_data = self.load_preset(preset_id, user_id)
                                    if include_metadata:
                                        presets.append(preset_data)
                                    else:
                                        # Only include basic info without full configuration
                                        presets.append({
                                            "id": preset_id,
                                            "name": preset_data.get("name", "Unnamed Preset"),
                                            "tags": preset_data.get("tags", []),
                                            "metadata": preset_data.get("metadata", {})
                                        })
                                except Exception as e:
                                    logger.warning(f"Error loading preset '{preset_id}': {str(e)}")
                
                # Sort presets by last modified date (newest first)
                presets.sort(key=lambda p: p.get("metadata", {}).get("last_modified", ""), reverse=True)
                
                self._preset_list_cache = presets
                self._preset_list_dirty = False
        
        return self._preset_list_cache.copy()
    
    def search_presets(self, query: str = None, tags: List[str] = None, 
                      user_id: str = None, include_public: bool = True) -> List[Dict[str, Any]]:
        """
        Search for presets matching the given criteria.
        
        Args:
            query (str, optional): Search term to match against preset names or descriptions.
            tags (List[str], optional): List of tags to match against preset tags.
            user_id (str, optional): Filter presets for a specific user.
            include_public (bool, optional): Whether to include public presets in search.
                                           Defaults to True.
        
        Returns:
            List[Dict[str, Any]]: List of matching presets.
        """
        presets = self.list_presets(user_id, include_public)
        
        # Filter by query
        if query:
            query = query.lower()
            presets = [
                p for p in presets 
                if (query in p.get("name", "").lower() or 
                    query in p.get("description", "").lower())
            ]
        
        # Filter by tags
        if tags:
            tags = [tag.lower() for tag in tags]
            presets = [
                p for p in presets 
                if any(tag.lower() in [t.lower() for t in p.get("tags", [])] 
                       for tag in tags)
            ]
        
        return presets
    
    def export_preset(self, preset_id: str, export_path: str, 
                     user_id: str = None) -> bool:
        """
        Export a preset to an external file.
        

