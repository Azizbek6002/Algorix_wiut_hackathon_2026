# Fixed CCTV Scene Geometry Specification (camera.md)

Bu fayl tashkilotchilar tomonidan beriladigan yoki videoga qarab bir marta kalibratsiya qilinadigan scene geometriyasi.
Koordinatalar `[X, Y]` piksellarda, tasvir o'lchami: 1920x1080.

---

## 1. Scene Overview
- **Resolution**: 1920 x 1080
- **Nominal FPS**: 30.0
- **Camera Type**: Fixed Overhead Intersection CCTV
- **Description**: 4-yo'lakli shahar chorrahasi (2 ta to'g'ri, 1 ta burilish, 1 ta qarama-qarshi yo'nalish), piyodalar o'tish yo'lagi va stop-line.

---

## 2. Road Polygons (Carriageway)
Avtomobillar harakatlanishi mumkin bo'lgan umumiy qatnov qismi:

```json
{
  "carriageway": [
    [240, 1080],
    [580, 420],
    [1340, 420],
    [1680, 1080]
  ]
}
```

---

## 3. Lanes (Yo'laklar)
Har bir yo'lak poligoni va kutilgan ruxsat etilgan harakat vektori (`expected_direction` [dx, dy]):

```json
[
  {
    "id": "lane_south_1",
    "name": "To'g'ri yo'lak (O'ng)",
    "type": "straight",
    "expected_direction": [0.0, 1.0],
    "polygon": [
      [960, 420],
      [1340, 420],
      [1680, 1080],
      [1180, 1080]
    ]
  },
  {
    "id": "lane_south_2",
    "name": "To'g'ri yo'lak (Chap)",
    "type": "straight",
    "expected_direction": [0.0, 1.0],
    "polygon": [
      [760, 420],
      [960, 420],
      [1180, 1080],
      [720, 1080]
    ]
  },
  {
    "id": "lane_north_oncoming",
    "name": "Qarama-qarshi yo'lak",
    "type": "oncoming",
    "expected_direction": [0.0, -1.0],
    "polygon": [
      [580, 420],
      [760, 420],
      [720, 1080],
      [240, 1080]
    ]
  }
]
```

---

## 4. Solid Lines (Uzluksiz chiziqlar)
Bosib o'tish taqiqlangan chiziqlar segmentlari (`solid_line_crossing` uchun):

```json
[
  {
    "id": "solid_divider_center",
    "segment": [
      [760, 420],
      [720, 1080]
    ]
  }
]
```

---

## 5. Stop Lines
Svetofor oldidagi to'xtash chizig'i segmentlari (`red_light` va `stop_line` hodisalari uchun):

```json
[
  {
    "id": "stop_line_main",
    "segment": [
      [760, 680],
      [1450, 680]
    ],
    "linked_lanes": ["lane_south_1", "lane_south_2"]
  }
]
```

---

## 6. Crossing Area (Piyodalar o'tish joyi)
`jaywalking` va `failure_to_yield` aniqlash uchun ruxsat etilgan piyodalar hududi (Zebra):

```json
[
  {
    "id": "crosswalk_south",
    "polygon": [
      [680, 720],
      [1500, 720],
      [1540, 800],
      [640, 800]
    ]
  }
]
```

---

## 7. Traffic Light ROI
Svetofor boshqaruv qutisi hududi (agar kadrlarda mavjud bo'lsa):

```json
{
  "has_traffic_light": true,
  "roi": [1480, 220, 60, 140]
}
```
