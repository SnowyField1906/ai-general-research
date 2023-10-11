import { useEffect, useState } from "react";

function App() {
  const [degree, setDegree] = useState(1);
  const [points, setPoints] = useState([[null, null]]);

  const setPoint = (e, index, isX) => {
    const newPoints = points.map((point, i) => {
      if (i === index) {
        return [isX ? e.target.value : point[0], isX ? point[1] : e.target.value];
      }
      return point;
    });

    if (newPoints[index][0] === "" && newPoints[index][1] === "") {
      newPoints.splice(index, 1);
    }

    setPoints(newPoints);
  };

  useEffect(() => {
    if (points[points.length - 1][0] != null || points[points.length - 1][1] != null) {
      setPoints([...points, [null, null]]);
    }

  }, [points]);

  const submit = () => {
    points.pop();
    const data = {
      degree: degree,
      points: points
    };
  }

  const isDisabled = () => {
    let isDisabled = false;

    if (points.length < 2) {
      return true
    }
    points.forEach((point, index) => {
      if ((point[0] == "" || point[0] == null || point[1] == "" || point[1] == null) && index !== points.length - 1) {
        isDisabled = true;
      }
    });
    return isDisabled;
  }

  return (
    <div className="py-5 px-3">
      <h1 className="text-center text-5xl font-semibold">Least Squares Computer</h1>
      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col items-center justify-center py-10">
          <p className="text-2xl text-center font-semibold my-5">Points</p>
          <div className="flex flex-col gap-2">
            {
              points.map((point, index) => (
                <div key={index} className="flex gap-2">
                  <input className="border-2 border-gray-600 rounded-xl px-2 py-1 font-semibold"
                    type="number" value={point[0] == null ? "" : point[0]} onChange={e => setPoint(e, index, true)} />
                  <input className="border-2 border-gray-600 rounded-xl px-2 py-1 font-semibold"
                    type="number" value={point[1] == null ? "" : point[1]} onChange={e => setPoint(e, index, false)} />
                </div>
              ))
            }
          </div>
          <p className="text-2xl text-center font-semibold my-5">Degree</p>
          <div className="flex gap-2">
            <input className="border-2 border-gray-600 rounded-xl px-2 py-1 font-semibold"
              type="number" value={degree} onChange={e => setDegree(e.target.value)} />
          </div>
          <button className="my-5 bg-blue-500 text-white rounded-xl py-2 px-6 hover:bg-blue-600 disabled:bg-gray-400"
            disabled={isDisabled()}
            onClick={submit}>Submit</button>
        </div>
      </div>
    </div>
  );
}

export default App;
