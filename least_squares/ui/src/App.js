import { useEffect, useState } from "react";
import axios from "axios";

function App() {
  const [degree, setDegree] = useState(1);
  const [points, setPoints] = useState([[null, null]]);
  const [result, setResult] = useState(null);

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

  const submit = async () => {

    let data_points = points.map(point => {
      return [parseFloat(point[0]), parseFloat(point[1])];
    })

    data_points.pop();

    const data = {
      degree: +degree,
      data_points: data_points
    };

    console.log(data)

    await axios.post(
      '/least-squares',
      data,
      {
        headers: {
          'Content-Type': 'application/json',
        }
      }
    ).then(res => setResult(res.data));
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
        <div className="flex flex-col items-center justify-center py-10">
          <p className="text-2xl text-center font-semibold my-5">Result</p>
          <div className="flex flex-col gap-2">
            <img src={`data:image/png;base64,${result?.img}`} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
