import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config_manager import ConfigManager
from src.preset.preset_model import Preset, PresetCategory
from src.preset.preset_repository import PresetRepository
from src.ui.integration_framework import UIIntegrationFramework
from src.ui.visualization_engine import VisualizationEngine


class PresetManagerUI:
    """
    PresetManagerUI handles the user interface and interactions for preset management.
    This class provides functionality for displaying, creating, editing, and deleting presets,
    as well as exporting and importing presets for sharing.
    """

    def __init__(self, ui_framework: UIIntegrationFramework, 
                 visualization_engine: VisualizationEngine,
                 preset_repository: PresetRepository,
                 config_manager: ConfigManager):
        """
        Initialize the PresetManagerUI with required dependencies.
        
        Args:
            ui_framework: The UI integration framework for interacting with the application UI
            visualization_engine: The engine for rendering visualizations
            preset_repository: Repository for storing and retrieving presets
            config_manager: Manager for application configuration
        """
        self.logger = logging.getLogger(__name__)
        self.ui_framework = ui_framework
        self.visualization_engine = visualization_engine
        self.preset_repository = preset_repository
        self.config_manager = config_manager
        
        # Event callbacks
        self.on_preset_selection_changed = None
        self.on_preset_applied = None
        self.on_preset_saved = None
        
        # UI state
        self.current_preset = None
        self.filter_category = None
        self.search_query = ""
        self.is_edit_mode = False
        
        self.logger.info("Preset Manager UI initialized")

    def register_ui_components(self):
        """
        Register all UI components with the UI framework.
        This method sets up the preset management panel and all its interactive elements.
        """
        # Create the main preset management panel
        preset_panel = self.ui_framework.create_panel(
            title="Preset Management",
            id="preset_manager_panel",
            collapsible=True,
            position="right",
            width=320
        )
        
        # Add search and filter controls
        self._add_search_and_filter_controls(preset_panel)
        
        # Add preset list display
        self._add_preset_list_view(preset_panel)
        
        # Add action buttons
        self._add_action_buttons(preset_panel)
        
        # Add editor panel (initially hidden)
        self._create_preset_editor_panel()
        
        self.logger.debug("Preset Manager UI components registered")
        
    def _add_search_and_filter_controls(self, panel):
        """Add search box and category filter dropdown to the panel"""
        # Search box for filtering presets by name
        self.ui_framework.add_text_input(
            parent=panel,
            id="preset_search",
            label="Search Presets",
            placeholder="Enter preset name...",
            on_change=self._handle_search_change
        )
        
        # Category filter dropdown
        categories = [{"value": cat.value, "label": cat.value.capitalize()} 
                     for cat in PresetCategory]
        categories.insert(0, {"value": "", "label": "All Categories"})
        
        self.ui_framework.add_dropdown(
            parent=panel,
            id="preset_category_filter",
            label="Filter by Category",
            options=categories,
            default_value="",
            on_change=self._handle_category_filter_change
        )

    def _add_preset_list_view(self, panel):
        """Add the list view for displaying available presets"""
        self.preset_list_container = self.ui_framework.add_container(
            parent=panel,
            id="preset_list_container",
            height=400,
            scrollable=True
        )
        
        # Initial population of the preset list
        self.refresh_preset_list()

    def _add_action_buttons(self, panel):
        """Add action buttons for managing presets"""
        button_container = self.ui_framework.add_container(
            parent=panel,
            id="preset_action_buttons",
            layout="horizontal",
            spacing=10
        )
        
        # Create New Preset button
        self.ui_framework.add_button(
            parent=button_container,
            id="btn_new_preset",
            label="New Preset",
            icon="plus",
            on_click=self._handle_new_preset
        )
        
        # Import Preset button
        self.ui_framework.add_button(
            parent=button_container,
            id="btn_import_preset",
            label="Import",
            icon="download",
            on_click=self._handle_import_preset
        )
        
        # Apply Preset button (disabled until selection)
        self.ui_framework.add_button(
            parent=button_container,
            id="btn_apply_preset",
            label="Apply",
            icon="play",
            enabled=False,
            on_click=self._handle_apply_preset
        )

    def _create_preset_editor_panel(self):
        """Create the panel for editing preset details (initially hidden)"""
        editor_panel = self.ui_framework.create_panel(
            title="Edit Preset",
            id="preset_editor_panel",
            visible=False,
            modal=True,
            width=600,
            height=700
        )
        
        # Preset basic information
        self.ui_framework.add_text_input(
            parent=editor_panel,
            id="preset_name_input",
            label="Preset Name",
            placeholder="Enter preset name...",
            required=True
        )
        
        # Category selection
        categories = [{"value": cat.value, "label": cat.value.capitalize()} 
                     for cat in PresetCategory]
        
        self.ui_framework.add_dropdown(
            parent=editor_panel,
            id="preset_category_select",
            label="Category",
            options=categories,
            required=True
        )
        
        # Tags input
        self.ui_framework.add_tags_input(
            parent=editor_panel,
            id="preset_tags_input",
            label="Tags",
            placeholder="Add tags...",
            help_text="Add tags like 'relaxation', 'focus', etc."
        )
        
        # Configuration editor
        # This section will dynamically show configuration options based on the category
        self.config_editor_container = self.ui_framework.add_container(
            parent=editor_panel,
            id="config_editor_container"
        )
        
        # Action buttons
        button_container = self.ui_framework.add_container(
            parent=editor_panel,
            id="editor_action_buttons",
            layout="horizontal",
            spacing=10
        )
        
        self.ui_framework.add_button(
            parent=button_container,
            id="btn_save_preset",
            label="Save",
            primary=True,
            on_click=self._handle_save_preset
        )
        
        self.ui_framework.add_button(
            parent=button_container,
            id="btn_cancel_edit",
            label="Cancel",
            on_click=self._handle_cancel_edit
        )

    def refresh_preset_list(self):
        """
        Refresh the list of presets displayed in the UI based on current filters.
        """
        # Clear current list
        self.ui_framework.clear_container(self.preset_list_container)
        
        # Get filtered presets from repository
        presets = self.preset_repository.list_presets(
            category=self.filter_category,
            search_term=self.search_query
        )
        
        if not presets:
            # Show empty state
            self.ui_framework.add_text(
                parent=self.preset_list_container,
                text="No presets found. Create a new preset to get started.",
                style="italic"
            )
            return
        
        # Add each preset to the list
        for preset in presets:
            self._add_preset_item(preset)
            
        self.logger.debug(f"Refreshed preset list with {len(presets)} presets")

    def _add_preset_item(self, preset: Preset):
        """
        Add a single preset item to the preset list.
        
        Args:
            preset: The preset to add to the list
        """
        item = self.ui_framework.add_list_item(
            parent=self.preset_list_container,
            id=f"preset_item_{preset.id}",
            selectable=True,
            on_select=lambda: self._handle_preset_selection(preset)
        )
        
        # Add content to the list item
        content_container = self.ui_framework.add_container(
            parent=item,
            layout="vertical",
            padding=10
        )
        
        # Preset name and category
        header = self.ui_framework.add_container(
            parent=content_container,
            layout="horizontal",
            justify="space-between"
        )
        
        self.ui_framework.add_text(
            parent=header,
            text=preset.name,
            style="heading"
        )
        
        self.ui_framework.add_badge(
            parent=header,
            text=preset.category.capitalize(),
            color=self._get_category_color(preset.category)
        )
        
        # Tags display
        if preset.tags:
            tags_container = self.ui_framework.add_container(
                parent=content_container,
                layout="horizontal",
                wrap=True,
                gap=5,
                margin_top=5
            )
            
            for tag in preset.tags:
                self.ui_framework.add_badge(
                    parent=tags_container,
                    text=tag,
                    size="small",
                    color="gray"
                )
        
        # Action buttons for this preset
        actions = self.ui_framework.add_container(
            parent=content_container,
            layout="horizontal",
            justify="flex-end",
            gap=5,
            margin_top=10
        )
        
        self.ui_framework.add_button(
            parent=actions,
            icon="edit",
            tooltip="Edit preset",
            size="small",
            on_click=lambda: self._handle_edit_preset(preset)
        )
        
        self.ui_framework.add_button(
            parent=actions,
            icon="upload",
            tooltip="Export preset",
            size="small",
            on_click=lambda: self._handle_export_preset(preset)
        )
        
        self.ui_framework.add_button(
            parent=actions,
            icon="trash",
            tooltip="Delete preset",
            size="small",
            color="danger",
            on_click=lambda: self._handle_delete_preset(preset)
        )

    def _get_category_color(self, category: str) -> str:
        """Get the appropriate color for a category badge"""
        color_map = {
            "genre": "blue",
            "mood": "purple",
            "consciousness": "green",
            "neural": "orange",
            "sacred": "red"
        }
        return color_map.get(category.lower(), "gray")

    def _handle_search_change(self, value: str):
        """Handle changes to the search input"""
        self.search_query = value
        self.refresh_preset_list()

    def _handle_category_filter_change(self, value: str):
        """Handle changes to the category filter dropdown"""
        self.filter_category = value if value else None
        self.refresh_preset_list()

    def _handle_preset_selection(self, preset: Preset):
        """Handle selection of a preset from the list"""
        self.current_preset = preset
        
        # Enable the apply button
        self.ui_framework.set_enabled("btn_apply_preset", True)
        
        # Show preset details in a preview panel
        self._update_preset_preview(preset)
        
        # Trigger the selection changed event
        if self.on_preset_selection_changed:
            self.on_preset_selection_changed(preset)

    def _update_preset_preview(self, preset: Preset):
        """Update the preview panel with the selected preset's details"""
        # Implementation depends on the visualization capabilities
        # This could show a summary of the preset's effects on the beat
        try:
            # Create a visualization of the preset parameters
            self.visualization_engine.create_preset_visualization(
                preset=preset,
                container="preset_preview_container"
            )
        except Exception as e:
            self.logger.error(f"Error generating preset visualization: {e}")

    def _handle_new_preset(self):
        """Handle the 'New Preset' button click"""
        self.is_edit_mode = False
        self._open_preset_editor(None)

    def _handle_edit_preset(self, preset: Preset):
        """Handle editing an existing preset"""
        self.is_edit_mode = True
        self._open_preset_editor(preset)

    def _open_preset_editor(self, preset: Optional[Preset] = None):
        """Open the preset editor with the given preset or an empty form"""
        # Show the editor panel
        self.ui_framework.set_visible("preset_editor_panel", True)
        
        if preset:
            # Populate form with preset data
            self.ui_framework.set_value("preset_name_input", preset.name)
            self.ui_framework.set_value("preset_category_select", preset.category)
            self.ui_framework.set_value("preset_tags_input", preset.tags)
            
            # Update panel title
            self.ui_framework.set_panel_title("preset_editor_panel", "Edit Preset")
            
            # Load configuration based on preset category
            self._load_configuration_editor(preset.category, preset.config_data)
        else:
            # Clear form for new preset
            self.ui_framework.set_value("preset_name_input", "")
            self.ui_framework.set_value("preset_category_select", "")
            self.ui_framework.set_value("preset_tags_input", [])
            
            # Update panel title
            self.ui_framework.set_panel_title("preset_editor_panel", "Create New Preset")
            
            # Load empty configuration editor
            self._load_configuration_editor("", {})
        
        # Add a change event handler for the category dropdown
        self.ui_framework.on_change(
            "preset_category_select", 
            lambda value: self._load_configuration_editor(value, {})
        )

    def _load_configuration_editor(self, category: str, config_data: Dict[str, Any]):
        """
        Load the appropriate configuration editor based on preset category.
        
        Args:
            category: The category of the preset
            config_data: The configuration data to populate the editor with
        """
        # Clear the current configuration editor
        self.ui_framework.clear_container(self.config_editor_container)
        
        if not category:
            # Show a message to select a category first
            self.ui_framework.

