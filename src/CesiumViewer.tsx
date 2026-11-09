import { useEffect, useRef } from "react";
import {
  ArcType,
  CallbackProperty,
  Cartesian3,
  Color,
  CzmlDataSource,
  createOsmBuildingsAsync,
  Ellipsoid,
  IntersectionTests,
  Ion,
  JulianDate,
  Math as CesiumMath,
  PolygonHierarchy,
  PositionProperty,
  Quaternion,
  Ray,
  Terrain,
  Viewer,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

const ORBIT_CZML_URL = "/orbit.czml";
const CONE_HALF_ANGLE = CesiumMath.toRadians(10);
const FOOTPRINT_SEGMENTS = 180;

interface ConeState {
  apexDirection: Cartesian3; // unit vector pointing from the cone toward the spacecraft
  center: Cartesian3;
  length: number;
  footprint: Cartesian3[];
  closed: boolean;
}

// CZML can't orient a cone along a time-varying inertial vector, so the cone is built here
// from the spacecraft and pointing-target positions (both evaluated in the fixed frame).
function addPointingCone(dataSource: CzmlDataSource) {
  const spacecraft = dataSource.entities.getById("spacecraft");
  const target = dataSource.entities.getById("pointing_target");
  if (!spacecraft?.position || !target?.position) return;

  const sc = new Cartesian3();
  const tgt = new Cartesian3();
  const axis = new Cartesian3();
  const direction = new Cartesian3();
  const u = new Cartesian3();
  const v = new Cartesian3();
  const ray = new Ray();
  const cosA = Math.cos(CONE_HALF_ANGLE);
  const sinA = Math.sin(CONE_HALF_ANGLE);

  // Evaluated several times per frame by different properties, so cache by time.
  let cachedTime: JulianDate | undefined;
  let cachedState: ConeState | undefined;

  // The cone runs from the spacecraft to the farthest point where its surface meets the Earth.
  // The part that would sit below the surface is hidden by the globe's depth test.
  const computeState = (time: JulianDate | undefined): ConeState | undefined => {
    if (!time) return undefined;
    if (cachedTime && JulianDate.equals(cachedTime, time)) return cachedState;
    cachedTime = JulianDate.clone(time, cachedTime);
    cachedState = undefined;

    if (!spacecraft.position!.getValue(time, sc) || !target.position!.getValue(time, tgt)) {
      return undefined;
    }
    Cartesian3.subtract(tgt, sc, direction);
    const targetDistance = Cartesian3.magnitude(direction);
    if (targetDistance === 0) return undefined;
    Cartesian3.divideByScalar(direction, targetDistance, direction);
    const helper = Math.abs(direction.z) < 0.9 ? Cartesian3.UNIT_Z : Cartesian3.UNIT_X;
    Cartesian3.normalize(Cartesian3.cross(direction, helper, u), u);
    Cartesian3.cross(direction, u, v);

    // Footprint: where rays along the cone surface first hit the ellipsoid.
    const footprint: Cartesian3[] = [];
    let axialLength = 0;
    for (let i = 0; i <= FOOTPRINT_SEGMENTS; i++) {
      const phi = (2 * Math.PI * i) / FOOTPRINT_SEGMENTS;
      const c = Math.cos(phi) * sinA;
      const s = Math.sin(phi) * sinA;
      ray.origin = Cartesian3.clone(sc, ray.origin);
      ray.direction = new Cartesian3(
        cosA * direction.x + c * u.x + s * v.x,
        cosA * direction.y + c * u.y + s * v.y,
        cosA * direction.z + c * u.z + s * v.z,
      );
      const hit = IntersectionTests.rayEllipsoid(ray, Ellipsoid.WGS84);
      if (hit && hit.start > 0) {
        footprint.push(Ray.getPoint(ray, hit.start));
        axialLength = Math.max(axialLength, hit.start * cosA);
      }
    }

    // Cone misses the Earth entirely: fall back to ending at the pointing target.
    const length = axialLength > 0 ? axialLength : targetDistance;
    const center = Cartesian3.add(
      sc,
      Cartesian3.multiplyByScalar(direction, length / 2, new Cartesian3()),
      new Cartesian3(),
    );
    cachedState = {
      apexDirection: Cartesian3.negate(direction, new Cartesian3()),
      center,
      length,
      footprint,
      closed: footprint.length === FOOTPRINT_SEGMENTS + 1,
    };
    return cachedState;
  };

  // Apex of the cone sits at the spacecraft, so the cylinder's +Z end points back at it.
  dataSource.entities.add({
    id: "pointing_cone",
    name: "Pointing cone",
    position: new CallbackProperty((time) => computeState(time)?.center, false) as unknown as PositionProperty,
    orientation: new CallbackProperty(
      (time) => {
        const g = computeState(time);
        if (!g) return undefined;
        Cartesian3.cross(Cartesian3.UNIT_Z, g.apexDirection, axis);
        if (Cartesian3.magnitude(axis) < 1e-9) {
          return g.apexDirection.z > 0 ? Quaternion.IDENTITY : Quaternion.fromAxisAngle(Cartesian3.UNIT_X, Math.PI);
        }
        const angle = Math.acos(CesiumMath.clamp(g.apexDirection.z, -1, 1));
        return Quaternion.fromAxisAngle(Cartesian3.normalize(axis, axis), angle);
      },
      false,
    ),
    cylinder: {
      length: new CallbackProperty((time) => computeState(time)?.length ?? 1, false),
      topRadius: 0,
      bottomRadius: new CallbackProperty(
        (time) => (computeState(time)?.length ?? 0) * Math.tan(CONE_HALF_ANGLE),
        false,
      ),
      material: Color.RED.withAlpha(0.25),
      outline: true,
      outlineColor: Color.RED.withAlpha(0.6),
      numberOfVerticalLines: 0,
    },
  });

  dataSource.entities.add({
    id: "pointing_footprint",
    name: "Pointing cone footprint",
    polyline: {
      positions: new CallbackProperty((time) => computeState(time)?.footprint ?? [], false),
      width: 3,
      arcType: ArcType.NONE,
      material: Color.DARKRED,
    },
    polygon: {
      hierarchy: new CallbackProperty((time) => {
        const g = computeState(time);
        // Only fill when every ray hit the Earth; otherwise the outline is open and has no interior.
        return new PolygonHierarchy(g?.closed ? g.footprint.slice(0, -1) : []);
      }, false),
      show: new CallbackProperty((time) => computeState(time)?.closed ?? false, false),
      arcType: ArcType.GEODESIC,
      material: Color.DARKRED.withAlpha(0.35),
    },
  });
}

Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_ION_TOKEN ?? "";

export default function CesiumViewer() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const viewer = new Viewer(containerRef.current, {
      terrain: Terrain.fromWorldTerrain({
        requestVertexNormals: true,
        requestWaterMask: true,
      }),
    });
    viewer.scene.globe.enableLighting = true;
    viewer.scene.globe.depthTestAgainstTerrain = true;

    let destroyed = false;
    createOsmBuildingsAsync()
      .then((buildings) => {
        if (destroyed) return;
        viewer.scene.primitives.add(buildings);
      })
      .catch((error) => console.error("Failed to load OSM Buildings", error));

    // The CZML document's clock block sets the viewer clock when the data source is added.
    CzmlDataSource.load(ORBIT_CZML_URL)
      .then((dataSource) => {
        if (destroyed) return;
        viewer.dataSources.add(dataSource);
        addPointingCone(dataSource);
        viewer.clock.shouldAnimate = true;
      })
      .catch((error) => console.error("Failed to load orbit CZML", error));

    // Wide view of Earth so the whole orbit is visible.
    viewer.camera.setView({
      destination: Cartesian3.fromDegrees(-122.4194, 20, 20_000_000),
    });

    return () => {
      destroyed = true;
      viewer.destroy();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ position: "fixed", inset: 0, width: "100vw", height: "100vh" }}
    />
  );
}
