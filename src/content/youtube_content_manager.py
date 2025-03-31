#!/usr/bin/env python3
"""
YouTube Content Manager

This module provides essential functionality for YouTube channel automation
with sacred geometry timing integration for optimal content performance.
"""

import logging
import datetime
from typing import Dict, List, Optional, Any, Tuple
import os
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.http import MediaFileUpload
    HAS_YOUTUBE_API = True
except ImportError:
    HAS_YOUTUBE_API = False
    logging.warning("YouTube API libraries not installed. Limited functionality available.")

from ..utils.sacred_geometry_core import SacredGeometryCore


class YouTubeContentManager:
    """
    Manages YouTube content creation, uploading, and analytics with
    sacred geometry principles for optimal timing and performance.
    """
    
    # YouTube API scopes required for functionality
    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/youtube.force-ssl"
    ]
    
    def __init__(self, 
                 credentials_path: str = "config/youtube_credentials.json",
                 token_path: str = "config/youtube_token.json",
                 sacred_geometry: Optional[SacredGeometryCore] = None):
        """
        Initialize the YouTube Content Manager.
        
        Args:
            credentials_path: Path to the YouTube API credentials file
            token_path: Path to store the OAuth token
            sacred_geometry: Optional SacredGeometryCore instance for timing optimization
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.youtube = None
        self.sacred_geometry = sacred_geometry or SacredGeometryCore()
        self.logger = logging.getLogger(__name__)
        
        if HAS_YOUTUBE_API:
            self._authenticate()
        else:
            self.logger.warning("YouTube API libraries not available. Please install them to use full functionality.")
    
    def _authenticate(self) -> None:
        """
        Authenticate with the YouTube API using OAuth2.
        Creates or loads credentials and builds the YouTube API service.
        """
        credentials = None
        
        # Load existing token if available
        if os.path.exists(self.token_path):
            credentials = Credentials.from_authorized_user_info(
                json.loads(Path(self.token_path).read_text()), self.SCOPES)
        
        # If no credentials or they're invalid, authenticate
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    self.logger.error(f"Credentials file not found at {self.credentials_path}")
                    return
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                credentials = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_path, 'w') as token:
                token.write(credentials.to_json())
        
        # Build the YouTube API service
        self.youtube = build('youtube', 'v3', credentials=credentials)
        self.logger.info("Successfully authenticated with YouTube API")
    
    def calculate_sacred_upload_time(self, 
                                     base_date: Optional[datetime.datetime] = None, 
                                     consciousness_level: int = 7) -> datetime.datetime:
        """
        Calculate the optimal upload time based on sacred geometry principles.
        
        Args:
            base_date: Base date to calculate from, defaults to current time
            consciousness_level: Target consciousness level (1-10)
            
        Returns:
            A datetime object representing the optimal upload time
        """
        base_date = base_date or datetime.datetime.now()
        
        # Use golden ratio to calculate optimal hour of day (phi-based timing)
        phi = self.sacred_geometry.PHI
        golden_hour = int((24 * phi) % 24)
        
        # Calculate optimal day using Fibonacci sequence
        day_offset = self.sacred_geometry.fibonacci(consciousness_level % 12)
        optimal_date = base_date + datetime.timedelta(days=day_offset)
        
        # Adjust minutes based on Schumann resonance
        schumann_minutes = int(60 * (self.sacred_geometry.SCHUMANN_FREQ % 1))
        
        # Set optimized time
        return optimal_date.replace(
            hour=golden_hour,
            minute=schumann_minutes,
            second=0,
            microsecond=0
        )
    
    def upload_video(self, 
                     video_path: str, 
                     title: str, 
                     description: str, 
                     tags: List[str],
                     category_id: str = "10",  # Music category
                     privacy_status: str = "public",
                     scheduled_time: Optional[datetime.datetime] = None,
                     use_sacred_timing: bool = True,
                     consciousness_level: int = 7) -> Dict[str, Any]:
        """
        Upload a video to YouTube with optimized metadata.
        
        Args:
            video_path: Path to the video file
            title: Video title
            description: Video description  
            tags: List of video tags
            category_id: YouTube category ID
            privacy_status: Privacy status (public, private, unlisted)
            scheduled_time: Specific scheduled time (overrides sacred timing)
            use_sacred_timing: Whether to use sacred geometry for timing
            consciousness_level: Target consciousness level for timing (1-10)
            
        Returns:
            Dictionary with upload response and video details
        """
        if not HAS_YOUTUBE_API or not self.youtube:
            self.logger.error("YouTube API not available. Cannot upload video.")
            return {"error": "YouTube API not available"}
            
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return {"error": f"Video file not found: {video_path}"}
            
        # Calculate optimal upload time based on sacred geometry if requested
        if use_sacred_timing and not scheduled_time:
            scheduled_time = self.calculate_sacred_upload_time(
                consciousness_level=consciousness_level
            )
            self.logger.info(f"Using sacred geometry optimal upload time: {scheduled_time}")
        
        # Enhance title and description with golden ratio text proportions
        if len(title) > 10:
            title_golden_length = int(len(title) / self.sacred_geometry.PHI)
            if title_golden_length > 5:
                # Insert a subtle pause at the golden ratio point
                title_parts = [title[:title_golden_length], title[title_golden_length:]]
                title = f"{title_parts[0]} | {title_parts[1]}"
        
        # Prepare metadata with sacred geometry optimizations
        video_metadata = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category_id
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False
            }
        }
        
        # Add scheduled publish time if specified
        if scheduled_time:
            # Convert to RFC 3339 format required by YouTube API
            scheduled_time_rfc = scheduled_time.isoformat() + "Z"
            video_metadata["status"]["publishAt"] = scheduled_time_rfc
        
        # Upload the video
        try:
            media = MediaFileUpload(video_path, 
                                    mimetype="application/octet-stream", 
                                    resumable=True)
            
            upload_request = self.youtube.videos().insert(
                part=",".join(video_metadata.keys()),
                body=video_metadata,
                media_body=media
            )
            
            self.logger.info(f"Beginning upload of video: {title}")
            response = upload_request.execute()
            
            video_id = response.get("id")
            if video_id:
                self.logger.info(f"Successfully uploaded video ID: {video_id}")
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                return {
                    "success": True,
                    "video_id": video_id,
                    "video_url": video_url,
                    "scheduled_time": scheduled_time.isoformat() if scheduled_time else None,
                    "sacred_timing_used": use_sacred_timing,
                    "response": response
                }
            else:
                self.logger.error("Failed to get video ID from upload response")
                return {"success": False, "error": "No video ID in response", "response": response}
                
        except Exception as e:
            self.logger.error(f"Error uploading video: {str(e)}")
            return {"success": False, "error": str(e)}

    def schedule_content(self, 
                         videos: List[Tuple[str, str, str, List[str]]],
                         base_date: Optional[datetime.datetime] = None,
                         days_between: int = 3,
                         golden_sequence: bool = True) -> List[Dict[str, Any]]:
        """
        Schedule multiple videos based on sacred geometry timing principles.
        
        Args:
            videos: List of tuples (video_path, title, description, tags)
            base_date: Starting date for scheduling
            days_between: Base number of days between uploads
            golden_sequence: Whether to use Fibonacci sequence for spacing
            
        Returns:
            List of scheduled upload details
        """
        base_date = base_date or datetime.datetime.now()
        results = []
        
        for i, (video_path, title, description, tags) in enumerate(videos):
            # Calculate days offset using Fibonacci sequence if requested
            if golden_sequence:
                # Use modulo to keep values reasonable while maintaining pattern
                fib_index = (i % 7) + 1  # Use first 7 Fibonacci numbers
                days_offset = self.sacred_geometry.fibonacci(fib_index) * days_between
            else:
                days_offset = i * days_between
                
            # Calculate scheduled date with sacred geometry
            scheduled_date = base_date + datetime.timedelta(days=days_offset)
            consciousness_level = ((i % 5) + 5)  # Vary between 5-9 for diversity
            
            upload_time = self.calculate_sacred_upload_time(
                base_date=scheduled_date,
                consciousness_level=consciousness_level
            )
            
            # Schedule the upload
            result = self.upload_video(
                video_path=video_path,
                title=title,
                description=description,
                tags=tags,
                scheduled_time=upload_time,
                use_sacred_timing=False  # We've already calculated the time
            )
            
            results.append({
                "video_info": {"path": video_path, "title": title},
                "scheduled_time": upload_time,
                "consciousness_level": consciousness_level,
                "days_offset": days_offset,
                "upload_result": result
            })
            
        return results
            
    def get_analytics(self, 
                     video_id: Optional[str] = None,
                     metrics: List[str] = None,
                     start_date: Optional[datetime.datetime] = None,
                     end_date: Optional[datetime.datetime] = None,
                     ) -> Dict[str, Any]:
        """
        Retrieve analytics data for videos, optimized for sacred geometry analysis.
        
        Args:
            video_id: Specific video ID or None for channel analytics
            metrics: List of metrics to retrieve (views, likes, etc.)
            start_date: Start date for analytics
            end_date: End date for analytics
            
        Returns:
            Dictionary with analytics data and sacred geometry correlation
        """
        if not HAS_YOUTUBE_API or not self.youtube:
            self.logger.error("YouTube API not available. Cannot retrieve analytics.")
            return {"error": "YouTube API not available"}
            
        # Default metrics if none provided
        metrics = metrics or ["views", "likes", "comments", "subscribersGained"]
        
        # Default dates if none provided
        end_date = end_date or datetime.datetime.now()
        start_date = start_date or (end_date - datetime.timedelta(days=30))
        
        # Format dates for YouTube API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        try:
            # Build analytics request
            analytics_request = self.youtube.reports().query(
                ids="channel==MINE",
                startDate=start_date_str,
                endDate=end_date_str,
                metrics=",".join(metrics),
                dimensions="day",
                filters=f"video=={video_id}" if video_id else ""
            )
            
            response = analytics_request.execute()
            
            # Analyze data with sacred geometry patterns
            # Here we would look for correlations between performance and sacred timing
            data = response.get("rows", [])
            sacred_analysis = self._analyze_performance_with_sacred_geometry(data, metrics)
            
            return {
                "success": True,
                "raw_data": response,
                "sacred_geometry_analysis": sacred_analysis,
                "optimal_future_days": self._predict_optimal_days(data)
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving analytics: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _analyze_performance_with_sacred_geometry(self, 
                                                data: List, 
                                                metrics: List[str]) -> Dict[str, Any]:
        """
        Analyze performance data using sacred geometry principles.
        
        Args:
            data: Analytics data rows
            metrics: Metrics included in the data
            
        Returns:
            Dictionary with sacred geometry analysis
        """
        # Placeholder for actual implementation
        # In a full implementation, this would:
        # 1. Identify performance peaks and relate to phi-based time points
        # 2. Calculate correlation with Fibonacci day patterns
        # 3. Identify golden ratio points in the performance curve
        
        return {
            "phi_correlation": 0.78,  # Placeholder for actual calculation
            "fibonacci_day_performance": {
                "1": {"performance": "baseline"},
                "2": {"performance": "+5%"},
                "3": {"performance": "+8%"},
                "5": {"performance": "+13%"},
                "8": {"performance": "+21%"}
            },
            "schumann_resonance_correlation": 0.63  # Placeholder
        }
    
    def _predict_optimal_days(self, data: List) -> List[datetime.datetime]:
        """
        Predict optimal future upload days based on past performance and sacred geometry.
        
        Args:
            data: Analytics data rows
            
        Returns:
            List of optimal future upload datetime objects
        """
        # In a full implementation, this would analyze patterns and predict
        # future optimal upload times based on sacred geometry principles
        
        # Placeholder implementation
        today = datetime.datetime.now()
        optimal_days

