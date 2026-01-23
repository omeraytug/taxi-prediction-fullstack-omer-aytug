import streamlit as st
import pandas as pd
import httpx
from typing import Optional, Tuple
from pathlib import Path
import os

# Import folium for map visualization
try:
    import folium
    from streamlit_folium import st_folium
    import polyline
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in project root (go up from frontend/app.py to project root)
    project_root = Path(__file__).parent.parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Also try current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env
    pass

# API configuration
API_BASE_URL = "http://127.0.0.1:8000"
GOOGLE_MAPS_API_URL = "https://maps.googleapis.com/maps/api/directions/json"

# Get Google Maps API key from environment variable
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

def get_prediction(input_data: dict) -> Optional[dict]:
    """Call the prediction API endpoint."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/predict/",
            json=input_data,
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Error connecting to API: {str(e)}")
        st.info("Make sure the FastAPI server is running on http://127.0.0.1:8000")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.status_code} - {e.response.text}")
        return None

def get_taxi_data() -> Optional[pd.DataFrame]:
    """Fetch taxi data from the API."""
    try:
        response = httpx.get(f"{API_BASE_URL}/taxi/", timeout=10.0)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except httpx.RequestError:
        return None
    except httpx.HTTPStatusError:
        return None

def get_route_info(origin: str, destination: str, api_key: str) -> Optional[Tuple[float, float, dict]]:
    """
    Get route distance, duration, and coordinates from Google Maps Directions API.
    
    Returns:
        Tuple of (distance_km, duration_minutes, route_data) or None if error
        route_data contains: origin_coords, destination_coords, overview_polyline
    """
    if not api_key:
        st.error("Google Maps API key is required for this feature.")
        return None
    
    try:
        params = {
            "origin": origin,
            "destination": destination,
            "key": api_key,
            "units": "metric"  # Get distance in kilometers
        }
        
        response = httpx.get(GOOGLE_MAPS_API_URL, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] != "OK":
            error_msg = data.get('error_message', data['status'])
            st.error(f"Google Maps API error: {error_msg}")
            if data["status"] == "REQUEST_DENIED":
                st.info("💡 Check if your API key is valid and the Directions API is enabled.")
            return None
        
        if not data.get("routes"):
            st.error("No route found between the specified locations.")
            return None
        
        # Get the first route
        route = data["routes"][0]
        leg = route["legs"][0]
        
        # Extract distance in km and duration in minutes
        distance_km = leg["distance"]["value"] / 1000  # Convert meters to km
        duration_minutes = leg["duration"]["value"] / 60  # Convert seconds to minutes
        
        # Extract coordinates for map display
        origin_coords = {
            "lat": leg["start_location"]["lat"],
            "lng": leg["start_location"]["lng"]
        }
        destination_coords = {
            "lat": leg["end_location"]["lat"],
            "lng": leg["end_location"]["lng"]
        }
        
        # Get route polyline for displaying the route
        overview_polyline = route.get("overview_polyline", {}).get("points", "")
        
        route_data = {
            "origin_coords": origin_coords,
            "destination_coords": destination_coords,
            "overview_polyline": overview_polyline,
            "origin_address": leg["start_address"],
            "destination_address": leg["end_address"]
        }
        
        return distance_km, duration_minutes, route_data
        
    except httpx.RequestError as e:
        st.error(f"Error connecting to Google Maps API: {str(e)}")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"Google Maps API error: {e.response.status_code}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Unexpected response format from Google Maps API: {str(e)}")
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Taxi Price Prediction",
        page_icon="🚕",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-box {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
            border: 2px solid #1f77b4;
            text-align: center;
            margin: 2rem 0;
        }
        .prediction-box h2 {
            color: #2c3e50;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .prediction-price {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
        }
        .stButton>button {
            width: 100%;
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚕 Taxi Price Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a page:",
            ["Prediction", "View Data"],
            index=0
        )
        st.markdown("---")
        st.header("Settings")
        
        # Load Google Maps API key from .env file only (no user input)
        google_api_key = DEFAULT_GOOGLE_API_KEY
        
        if google_api_key:
            st.success("✓ Google Maps API key loaded from `.env` file")
            st.info("📍 Location-based input is **enabled**")
        else:
            st.warning("⚠ Google Maps API key not found in `.env` file")
            st.info("💡 **Manual Input** mode is always available - you can enter distance and duration directly.")
            with st.expander("ℹ️ How to configure Google Maps API Key (Optional)"):
                st.markdown("""
                **This is optional!** The app works perfectly fine without it using Manual Input mode.
                
                To enable location-based input:
                1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                2. Create a new project or select an existing one
                3. Enable the **Directions API**
                4. Go to **APIs & Services > Credentials**
                5. Create an **API Key**
                6. Add it to your `.env` file as: `GOOGLE_MAPS_API_KEY=your_key_here`
                
                **Note:** The Directions API has usage limits. Check Google's pricing.
                """)
        
        st.markdown("---")
        st.info("**API Status**\n\nMake sure the FastAPI server is running on:\n`http://127.0.0.1:8000`")
    
    if page == "Prediction":
        st.header("Predict Taxi Trip Price")
        st.markdown("Enter the trip details below to get a price prediction.")
        
        # Input method selection
        # Only show Google Maps option if API key is available
        input_options = ["✏️ Manual Input"]
        if google_api_key:
            input_options.insert(0, "📍 Use Locations (Google Maps)")
        
        input_method = st.radio(
            "Choose input method:",
            input_options,
            horizontal=True,
            help="Select whether to use location names or enter distance/duration manually"
        )
        
        st.markdown("---")
        
        if input_method == "📍 Use Locations (Google Maps)":
            # Location-based input
            col1, col2 = st.columns(2)
            
            with col1:
                origin = st.text_input(
                    "📍 Origin",
                    placeholder="e.g., Stockholm Central Station, Stockholm",
                    help="Enter the starting location (address, city, or landmark)"
                )
            
            with col2:
                destination = st.text_input(
                    "🎯 Destination",
                    placeholder="e.g., Arlanda Airport, Stockholm",
                    help="Enter the destination location (address, city, or landmark)"
                )
            
            # Store origin/destination in session state for later use
            if origin:
                st.session_state['origin'] = origin
            if destination:
                st.session_state['destination'] = destination
            
            # Fetch route button
            route_fetched = False
            if st.button("🗺️ Get Route Info from Google Maps", type="primary", use_container_width=True):
                if not google_api_key:
                    st.error("❌ Google Maps API key is not configured. Please add `GOOGLE_MAPS_API_KEY` to your `.env` file to use this feature.")
                    st.info("💡 You can still use **Manual Input** mode to enter distance and duration directly.")
                elif not origin or not destination:
                    st.error("Please enter both origin and destination.")
                else:
                    with st.spinner("Fetching route information from Google Maps..."):
                        route_info = get_route_info(origin, destination, google_api_key)
                    
                    if route_info:
                        distance_km, duration_minutes, route_data = route_info
                        st.session_state['trip_distance'] = distance_km
                        st.session_state['trip_duration'] = duration_minutes
                        st.session_state['origin'] = origin
                        st.session_state['destination'] = destination
                        st.session_state['route_data'] = route_data
                        route_fetched = True
                        st.success(f"✅ Route found! Distance: {distance_km:.2f} km, Duration: {duration_minutes:.1f} minutes")
            
            # Display fetched route info if available
            if 'trip_distance' in st.session_state and 'trip_duration' in st.session_state:
                st.info(f"📊 **Current Route:** {st.session_state['trip_distance']:.2f} km, {st.session_state['trip_duration']:.1f} minutes")
                trip_distance = st.session_state['trip_distance']
                trip_duration = st.session_state['trip_duration']
                
                # Display map if route data is available
                if 'route_data' in st.session_state:
                    route_data = st.session_state['route_data']
                    st.markdown("### 🗺️ Route Map")
                    
                    if FOLIUM_AVAILABLE and route_data.get('overview_polyline'):
                        # Decode the polyline to get route coordinates
                        try:
                            route_coords = polyline.decode(route_data['overview_polyline'])
                            # Convert to (lat, lng) format for folium
                            route_coords = [(lat, lng) for lat, lng in route_coords]
                            
                            # Calculate center point for map
                            center_lat = (route_data['origin_coords']['lat'] + route_data['destination_coords']['lat']) / 2
                            center_lng = (route_data['origin_coords']['lng'] + route_data['destination_coords']['lng']) / 2
                            
                            # Create folium map
                            m = folium.Map(
                                location=[center_lat, center_lng],
                                zoom_start=10,
                                tiles='OpenStreetMap'
                            )
                            
                            # Add route polyline
                            folium.PolyLine(
                                route_coords,
                                color='blue',
                                weight=5,
                                opacity=0.7,
                                popup='Route'
                            ).add_to(m)
                            
                            # Add origin marker
                            folium.Marker(
                                location=[route_data['origin_coords']['lat'], route_data['origin_coords']['lng']],
                                popup=folium.Popup(f"<b>📍 Origin</b><br>{route_data.get('origin_address', 'N/A')}", max_width=300),
                                icon=folium.Icon(color='green', icon='info-sign')
                            ).add_to(m)
                            
                            # Add destination marker
                            folium.Marker(
                                location=[route_data['destination_coords']['lat'], route_data['destination_coords']['lng']],
                                popup=folium.Popup(f"<b>🎯 Destination</b><br>{route_data.get('destination_address', 'N/A')}", max_width=300),
                                icon=folium.Icon(color='red', icon='info-sign')
                            ).add_to(m)
                            
                            # Display the map
                            st_folium(m, width=700, height=500)
                            
                        except Exception as e:
                            st.warning(f"Could not decode route polyline: {str(e)}")
                            # Fallback to simple map with just points
                            map_data = pd.DataFrame({
                                "lat": [
                                    route_data['origin_coords']['lat'],
                                    route_data['destination_coords']['lat']
                                ],
                                "lon": [
                                    route_data['origin_coords']['lng'],
                                    route_data['destination_coords']['lng']
                                ],
                                "location": ["Origin", "Destination"]
                            })
                            st.map(map_data, zoom=10)
                    else:
                        # Fallback to simple map if folium not available
                        map_data = pd.DataFrame({
                            "lat": [
                                route_data['origin_coords']['lat'],
                                route_data['destination_coords']['lat']
                            ],
                            "lon": [
                                route_data['origin_coords']['lng'],
                                route_data['destination_coords']['lng']
                            ],
                            "location": ["Origin", "Destination"]
                        })
                        st.map(map_data, zoom=10)
                        if not FOLIUM_AVAILABLE:
                            st.info("💡 Install folium and streamlit-folium to see the route path: `uv add folium streamlit-folium polyline`")
                    
                    # Show route details
                    with st.expander("📍 Route Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Origin:**")
                            st.write(route_data.get('origin_address', 'N/A'))
                            st.write(f"Coordinates: {route_data['origin_coords']['lat']:.6f}, {route_data['origin_coords']['lng']:.6f}")
                        with col2:
                            st.write("**Destination:**")
                            st.write(route_data.get('destination_address', 'N/A'))
                            st.write(f"Coordinates: {route_data['destination_coords']['lat']:.6f}, {route_data['destination_coords']['lng']:.6f}")
            else:
                trip_distance = None
                trip_duration = None
                st.info("👆 Click 'Get Route Info' to fetch distance and duration from Google Maps")
        
        else:
            # Manual input
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Trip Details")
                trip_distance = st.number_input(
                    "Trip Distance (km)",
                    min_value=0.1,
                    max_value=500.0,
                    value=15.5,
                    step=0.1,
                    help="Distance of the trip in kilometers"
                )
                
                trip_duration = st.number_input(
                    "Trip Duration (minutes)",
                    min_value=1.0,
                    max_value=300.0,
                    value=30.0,
                    step=1.0,
                    help="Expected duration of the trip in minutes"
                )
        
        # Conditions section (always shown)
        st.markdown("---")
        st.subheader("Trip Conditions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            time_of_day = st.selectbox(
                "Time of Day",
                options=["Morning", "Afternoon", "Evening", "Night"],
                index=2,
                help="Time of day when the trip will occur"
            )
        
        with col2:
            day_of_week = st.selectbox(
                "Day of Week",
                options=["Weekday", "Weekend"],
                index=1,
                help="Whether it's a weekday or weekend"
            )
        
        with col3:
            traffic_conditions = st.selectbox(
                "Traffic Conditions",
                options=["Low", "Medium", "High"],
                index=0,
                help="Expected traffic conditions"
            )
        
        with col4:
            weather = st.selectbox(
                "Weather",
                options=["Clear", "Rain", "Snow"],
                index=0,
                help="Weather conditions"
            )
        
        st.markdown("---")
        
        # Prediction button
        can_predict = trip_distance is not None and trip_duration is not None
        
        if not can_predict and input_method == "📍 Use Locations (Google Maps)":
            st.warning("⚠️ Please fetch route information first before predicting.")
        
        if st.button("🔮 Predict Price", type="primary", use_container_width=True, disabled=not can_predict):
            # Prepare input data
            input_data = {
                "trip_distance_km": trip_distance,
                "trip_duration_minutes": trip_duration,
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "traffic_conditions": traffic_conditions,
                "weather": weather
            }
            
            # Show loading spinner
            with st.spinner("Getting prediction from API..."):
                result = get_prediction(input_data)
            
            if result:
                # Display prediction
                predicted_price = result["predicted_price"]
                
                st.markdown("---")
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h2>Predicted Price</h2>
                        <div class="prediction-price">${predicted_price:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show input summary
                with st.expander("📋 View Input Summary"):
                    summary_col1, summary_col2 = st.columns(2)
                    with summary_col1:
                        st.write("**Trip Details:**")
                        st.write(f"- Distance: {trip_distance:.2f} km")
                        st.write(f"- Duration: {trip_duration:.1f} minutes")
                        if input_method == "📍 Use Locations (Google Maps)" and 'origin' in st.session_state:
                            st.write(f"- Origin: {st.session_state.get('origin', 'N/A')}")
                            st.write(f"- Destination: {st.session_state.get('destination', 'N/A')}")
                    with summary_col2:
                        st.write("**Conditions:**")
                        st.write(f"- Time: {time_of_day}")
                        st.write(f"- Day: {day_of_week}")
                        st.write(f"- Traffic: {traffic_conditions}")
                        st.write(f"- Weather: {weather}")
                
                # Show processed features (optional, for debugging/transparency)
                with st.expander("🔍 View Processed Features (Technical)"):
                    st.json(result["input_features"])
    
    elif page == "View Data":
        st.header("Taxi Trip Data")
        st.markdown("View the cleaned dataset used for training the model.")
        
        # Fetch data
        with st.spinner("Loading data from API..."):
            df = get_taxi_data()
        
        if df is not None and not df.empty:
            st.success(f"Loaded {len(df)} records")
            
            # Display basic statistics
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                if "trip_price" in df.columns:
                    st.metric("Avg Price", f"${df['trip_price'].mean():.2f}")
            with col4:
                if "trip_distance_km" in df.columns:
                    st.metric("Avg Distance", f"{df['trip_distance_km'].mean():.2f} km")
            
            st.markdown("---")
            
            # Data table
            st.subheader("Data Table")
            st.dataframe(df, use_container_width=True, height=400)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name="taxi_trip_data.csv",
                mime="text/csv"
            )
        else:
            st.error("Could not load data from API. Make sure the FastAPI server is running.")

if __name__ == "__main__":
    main()
