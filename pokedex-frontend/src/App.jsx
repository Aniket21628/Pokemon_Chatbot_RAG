import React, { useState } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      
      if (!res.ok) {
        throw new Error('Network response was not ok');
      }
      
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      setResponse('Error: Could not fetch response from server.');
    }
    setLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-400 via-blue-400 to-violet-400 flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* Decorative elements */}
      <div className="absolute top-10 left-10 w-20 h-20 bg-yellow-300 rounded-full opacity-20 animate-pulse"></div>
      <div className="absolute bottom-20 right-10 w-16 h-16 bg-blue-300 rounded-full opacity-20 animate-pulse delay-1000"></div>
      <div className="absolute top-1/3 right-1/4 w-12 h-12 bg-green-300 rounded-full opacity-20 animate-pulse delay-500"></div>
      
      <h1 className="text-5xl font-bold text-white mb-8 text-center drop-shadow-lg animate-bounce">
        Rotom Pokédex Assistant
      </h1>
      
      <div className="relative flex flex-col items-center">
        {/* Chat bubble */}
        {(response || loading) && (
          <div className="relative mb-4 max-w-xs sm:max-w-md animate-fadeIn">
            <div className="bg-white rounded-3xl p-4 shadow-2xl border-4 border-orange-300 relative">
              <p className="text-gray-800 text-sm sm:text-base leading-relaxed">
                {loading ? (
                  <span className="flex items-center">
                    <span className="animate-spin mr-2">⚡</span>
                    Bzzt! Processing your query...
                  </span>
                ) : response}
              </p>
              {/* Speech bubble tail */}
              <div className="absolute bottom-0 left-1/2 transform translate-y-full -translate-x-1/2">
                <div className="w-0 h-0 border-l-[15px] border-r-[15px] border-t-[20px] border-l-transparent border-r-transparent border-t-white"></div>
                <div className="w-0 h-0 border-l-[18px] border-r-[18px] border-t-[23px] border-l-transparent border-r-transparent border-t-orange-300 absolute -top-1 left-1/2 transform -translate-x-1/2"></div>
              </div>
            </div>
          </div>
        )}
        
        {/* Rotom Character */}
      <div className="relative mb-6 transform hover:scale-105 transition-transform duration-300">
        <div className="w-48 h-48 relative animate-float">
          {/* Main body - oval orange shape */}
          <div className="absolute top-8 left-1/2 transform -translate-x-1/2 w-28 h-32 bg-gradient-to-b from-orange-400 to-orange-600 rounded-full shadow-xl border-2 border-orange-700 relative overflow-hidden">
            {/* Body highlight */}
            <div className="absolute top-2 left-4 w-6 h-8 bg-orange-300 rounded-full opacity-60"></div>
            {/* Body shadow */}
            <div className="absolute bottom-2 left-2 right-2 h-4 bg-orange-700 rounded-full opacity-30"></div>
          </div>

          {/* Eyes - larger blue eyes */}
          <div className="absolute top-12 left-1/2 transform -translate-x-1/2 -translate-x-6 w-6 h-6 bg-gradient-to-b from-cyan-300 to-cyan-500 rounded-full border-2 border-cyan-700 shadow-lg">
            <div className="absolute top-1 left-1 w-3 h-3 bg-white rounded-full"></div>
            <div className="absolute top-0.5 left-0.5 w-2 h-2 bg-white rounded-full opacity-80"></div>
          </div>
          <div className="absolute top-12 left-1/2 transform -translate-x-1/2 translate-x-6 w-6 h-6 bg-gradient-to-b from-cyan-300 to-cyan-500 rounded-full border-2 border-cyan-700 shadow-lg">
            <div className="absolute top-1 left-1 w-3 h-3 bg-white rounded-full"></div>
            <div className="absolute top-0.5 left-0.5 w-2 h-2 bg-white rounded-full opacity-80"></div>
          </div>

          {/* Zigzag mouth */}
          <div className="absolute top-20 left-1/2 transform -translate-x-1/2">
            <div className="relative w-8 h-3">
              <div className="absolute top-0 left-0 w-0 h-0 border-l-2 border-r-2 border-b-3 border-l-transparent border-r-transparent border-b-gray-800"></div>
              <div className="absolute top-0 left-2 w-0 h-0 border-l-2 border-r-2 border-t-3 border-l-transparent border-r-transparent border-t-gray-800"></div>
              <div className="absolute top-0 left-4 w-0 h-0 border-l-2 border-r-2 border-b-3 border-l-transparent border-r-transparent border-b-gray-800"></div>
              <div className="absolute top-0 left-6 w-0 h-0 border-l-2 border-r-2 border-t-3 border-l-transparent border-r-transparent border-t-gray-800"></div>
            </div>
          </div>

          {/* Top spike/point */}
          <div className="absolute top-2 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-6 border-r-6 border-b-8 border-l-transparent border-r-transparent border-b-gradient-to-b border-b-red-500 shadow-lg"></div>

          {/* Left arm/appendage */}
          <div className="absolute top-16 left-4 w-8 h-8 bg-gradient-to-br from-orange-400 to-red-500 rounded-full border-2 border-red-600 shadow-lg transform -rotate-12">
            <div className="absolute top-1 left-1 w-2 h-2 bg-red-700 rounded-full opacity-60"></div>
          </div>

          {/* Right arm/appendage */}
          <div className="absolute top-16 right-4 w-8 h-8 bg-gradient-to-bl from-orange-400 to-red-500 rounded-full border-2 border-red-600 shadow-lg transform rotate-12">
            <div className="absolute top-1 right-1 w-2 h-2 bg-red-700 rounded-full opacity-60"></div>
          </div>

          {/* Bottom left appendage */}
          <div className="absolute bottom-8 left-8 w-7 h-7 bg-gradient-to-br from-orange-400 to-red-500 rounded-full border-2 border-red-600 shadow-lg transform -rotate-45">
            <div className="absolute top-1 left-1 w-2 h-2 bg-red-700 rounded-full opacity-60"></div>
          </div>

          {/* Bottom right appendage */}
          <div className="absolute bottom-8 right-8 w-7 h-7 bg-gradient-to-bl from-orange-400 to-red-500 rounded-full border-2 border-red-600 shadow-lg transform rotate-45">
            <div className="absolute top-1 right-1 w-2 h-2 bg-red-700 rounded-full opacity-60"></div>
          </div>

          {/* Electric sparks effect */}
          <div className="absolute top-4 right-2 w-1 h-1 bg-yellow-300 rounded-full animate-ping"></div>
          <div className="absolute bottom-4 left-2 w-1 h-1 bg-yellow-300 rounded-full animate-ping delay-500"></div>
        </div>
        
        {/* Input section */}
        <div className="w-full max-w-md bg-white/90 backdrop-blur-sm rounded-2xl shadow-2xl p-6 border-2 border-orange-300">
          <div className="flex flex-col">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask Rotom about any Pokémon..."
              className="p-3 mb-4 border-2 border-orange-200 rounded-xl text-gray-800 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:border-orange-400 placeholder-gray-500 transition-all duration-200"
            />
            <button
              onClick={handleSubmit}
              disabled={loading || !query.trim()}
              className="bg-gradient-to-r from-orange-500 to-red-500 text-white p-3 rounded-xl hover:from-orange-600 hover:to-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-semibold shadow-lg transform hover:scale-105 active:scale-95"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <span className="animate-spin mr-2">⚡</span>
                  Bzzt! Loading...
                </span>
              ) : (
                <span className="flex items-center justify-center">
                  <span className="mr-2">⚡</span>
                  Ask Rotom
                </span>
              )}
            </button>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-float {
          animation: float 1s ease-in-out infinite;
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
      `}</style>
    </div>
  </div>
  );
}

export default App;